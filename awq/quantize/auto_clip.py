import torch
import torch.nn as nn
from .quantizer import pseudo_quantize_tensor
import gc

__all__ = ["auto_clip_block"]


# weight quantization
@torch.no_grad()
def auto_clip_layer(
    w, input_feat, n_bit, q_config, n_grid=20, max_shrink=0.5, n_sample_token=512
):
    """
    自动调整权重矩阵的裁剪阈值，以在量化过程中最小化量化误差。
    具体而言，它通过遍历多个裁剪阈值，对权重进行裁剪和量化，并选择使量化后输出与原始输出误差最小的裁剪阈值。

    Args:
        w: 权重矩阵，形状为[co, ci]。
        input_feat: 输入特征，形状为[n_token, ci]。
        n_bit: 量化位数。
        q_config: 量化配置参数，通常包括`q_group_size`等。
        n_grid: 网格数量，用于裁剪阈值的搜索。
        max_shrink: 最大收缩比例，用于限定裁剪阈值的最小缩小比例。
        n_sample_token: 采样的token数量，用于减少计算量。
    """
    assert w.dim() == 2                                             # 确保权重矩阵是二维的
    org_w_shape = w.shape                                           # 记录原始权重矩阵的形状，以便后续恢复
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
    group_size = (                                                  # 确定量化组的大小，如果`q_group_size`大于0，则使用`q_group_size`，否则使用`w`的第二维度（即输入特征的维度）
        q_config["q_group_size"] if q_config["q_group_size"] > 0 else w.shape[1]
    )
    input_feat = input_feat.view(-1, input_feat.shape[-1])          # 将输入特征展平，形状为[n_token, ci]
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)     # 重塑输入特征，形状为[1, n_token, n_group, group size]
    input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]      # 采样输入特征，从`n_token`均匀采样`n_sample_token`个token，以减少计算量
    w = w.reshape(w.shape[0], 1, -1, group_size)                    # 重塑权重矩阵，形状为[co, 1, n_group, group size]

    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM # 确定输出通道批量大小，如果输出通道数能被256整除，则设置为256，否则设置为64
    assert w.shape[0] % oc_batch_size == 0                          # 确保输出通道数能被批量大小整除
    w_all = w                                                       # 记录所有的权重矩阵
    best_max_val_all = []                                           # 记录每个批次的最佳裁剪阈值

    for i_b in range(w.shape[0] // oc_batch_size):                  # 遍历输出通道批次
        w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]  # 获取当前批次的权重矩阵，形状为[oc_batch_size, 1, n_group, group size]

        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1   # 计算每组权重的绝对值最大值，形状为[oc_batch_size, 1, n_group, 1]

        best_max_val = org_max_val.clone()                          # 初始化最佳裁剪阈值
        min_errs = torch.ones_like(org_max_val) * 1e9               # 初始化最小误差张量
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group  # 计算原始输出，形状为[oc_batch_size, n_token, n_group]

        for i_s in range(int(max_shrink * n_grid)):                 # 遍历裁剪阈值的网格
            max_val = org_max_val * (1 - i_s / n_grid)              # 计算当前裁剪阈值，逐步减小裁剪比例
            min_val = -max_val                                      # 计算最小裁剪阈值
            cur_w = torch.clamp(w, min_val, max_val)                # 裁剪权重矩阵
            q_w = pseudo_quantize_tensor(cur_w, n_bit=n_bit, **q_config)    # 量化裁剪后的权重矩阵
            cur_out = (input_feat * q_w).sum(dim=-1)                # 计算量化后的输出，形状为[oc_batch_size, n_token, n_group]

            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)   # 计算量化后的输出与原始输出之间的均方误差
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs                           # 判断当前误差是否小于最小误差
            min_errs[cur_best_idx] = err[cur_best_idx]              # 更新最小误差
            best_max_val[cur_best_idx] = max_val[cur_best_idx]      # 更新最佳裁剪阈值
        best_max_val_all.append(best_max_val)                       # 记录当前批次的最佳裁剪阈值

    best_max_val = torch.cat(best_max_val_all, dim=0)               # 合并所有批次的最佳裁剪阈值

    del input_feat
    del org_out
    gc.collect()
    torch.cuda.empty_cache()
    return best_max_val.squeeze(1)


@torch.no_grad()
def auto_clip_block(module, w_bit, q_config, input_feat):
    """
    对一个模块中的所有线性层执行自动裁剪操作。
    具体来说，它首先筛选出模块中的所有线性层，然后对每个线性层（除去查询和键相关的层）应用`auto_clip_layer`函数，获取最佳裁剪阈值。

    Args:
        module: 待处理的模块，通常为模型中的一个子模块。
        w_bit: 权重的量化位数。
        q_config: 量化配置参数。
        input_feat: 输入特征，键为线性层的名称，值为输入特征张量。
    """
    named_linears = {                                               # 获取模块中的所有线性层
        name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)
    }

    clip_list = []                                                  # 用于存储每个线性层的最佳裁剪阈值
    for name in named_linears:                                      # 遍历所有线性层
        # due to qk bmm, it is hard to clip precisely
        if any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]):  # 跳过与查询和键相关的线性层
            continue
        named_linears[name].cuda()
        max_val = auto_clip_layer(                                  # 获取当前线性层的最佳裁剪阈值
            named_linears[name].weight, input_feat[name], n_bit=w_bit, q_config=q_config
        )
        clip_list.append((name, max_val))                           # 记录当前线性层的最佳裁剪阈值
        named_linears[name].cpu()
    return clip_list


@torch.no_grad()
def apply_clip(module, clip_list):
    """
    将预先计算好的裁剪阈值应用到指定模块的线性层权重上。
    具体来说，它根据`clip_list`中的裁剪阈值，裁剪每个指定的线性层的权重，使其在裁剪阈值范围内。

    Args:
        module: 待处理的模块，通常为模型中的一个子模块。
        clip_list: 裁剪阈值列表，每个元素为一个元组，包含线性层名称和最佳裁剪阈值。
    """
    from ..utils.module import get_op_by_name

    for name, max_val in clip_list:                                 # 遍历裁剪阈值列表
        layer = get_op_by_name(module, name)                        # 获取指定名称的线性层
        layer.cuda()
        max_val = max_val.to(layer.weight.device)
        org_shape = layer.weight.shape                              # 记录原始权重矩阵的形状，以便后续恢复
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)   # 将线性层的权重矩阵重塑为与裁剪阈值前两维匹配的形状，以便进行逐组裁剪
        layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)   # 裁剪线性层的权重矩阵
        layer.weight.data = layer.weight.data.reshape(org_shape)    # 恢复裁剪后的权重矩阵的形状
        layer.cpu()

import torch
import torch.nn as nn
from tqdm import tqdm
import gc
from .qmodule import ScaledActivation
from ..utils.module import set_op_by_name

from transformers.models.bloom.modeling_bloom import BloomBlock

EMBEDDING_KEYWORDS = ["embed"]
LM_HEAD_KEYWORDS = ["lm_head", "embed_out", "output"]


def scale_activations(module):
    """
    在模型的特定模块中将激活函数替换为带有缩放因子的`ScaledActivation`模块。
    这有助于在量化过程中动态调整激活值，以适应不同精度的表示。

    Args:
        module: 需要缩放激活的模块。
    """
    param = next(module.parameters())
    dtype = param.dtype
    device = param.device
    # 处理`BloomBlock`模块
    if isinstance(module, BloomBlock):
        if isinstance(module.mlp.gelu_impl, ScaledActivation):              # 如果`gelu_impl`已经是`ScaledActivation`模块，则直接返回
            return
        c = module.mlp.dense_h_to_4h.out_features                           # 获取全连接层`dense_h_to_4h`的输出特征数
        act = ScaledActivation(                                             # 创建`ScaledActivation`实例
            module.mlp.gelu_impl, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.gelu_impl", act)                        # 替换`gelu_impl`为`ScaledActivation`实例
    # 处理包含`mpblock`的模块
    elif "mptblock" in str(module.__class__.__name__).lower():
        if isinstance(module.ffn.act, ScaledActivation):                    # 如果`ffn.act`已经是`ScaledActivation`模块，则直接返回
            return
        c = module.ffn.up_proj.out_features                                 # 获取`up_proj`的输出特征数
        act = ScaledActivation(                                             # 创建`ScaledActivation`实例
            module.ffn.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "ffn.act", act)                              # 替换`ffn.act`为`ScaledActivation`实例
    # 处理包含`falcon`的模块
    elif "falcon" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):                    # 如果`mlp.act`已经是`ScaledActivation`模块，则直接返回
            return
        c = module.mlp.dense_h_to_4h.out_features                           # 获取`dense_h_to_4h`的输出特征数
        act = ScaledActivation(                                             # 创建`ScaledActivation`实例
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)                              # 替换`mlp.act`为`ScaledActivation`实例
    # 处理包含`bigcode`的模块
    elif "bigcode" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):                    # 如果`mlp.act`已经是`ScaledActivation`模块，则直接返回
            return
        c = module.mlp.c_proj.out_features                                  # 获取`c_proj`的输出特征数
        act = ScaledActivation(                                             # 创建`ScaledActivation`实例
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)                              # 替换`mlp.act`为`ScaledActivation`实例
    # 处理包含`neox`的模块
    elif "neox" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):                    # 如果`mlp.act`已经是`ScaledActivation`模块，则直接返回
            return
        c = module.mlp.dense_h_to_4h.out_features                           # 获取`dense_h_to_4h`的输出特征数
        act = ScaledActivation(                                             # 创建`ScaledActivation`实例
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)                              # 替换`mlp.act`为`ScaledActivation`实例


# core quantization method (simulated quantization)
def pseudo_quantize_tensor(
    w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    """
    对给定的权重张量进行模拟量化（伪量化）。
    其目的是将高精度的浮点数张量替换为低精度的整数表示，再恢复为浮点数，从而模拟量化过程对权重的影响。

    Args:
        w: 待量化的权重张量。
        n_bit: 量化的比特数，默认为8。
        zero_point: 是否使用零点量化，默认为True。
        q_group_size: 量化组大小，默认为-1（表示不分组）。
        inplace: 是否原地操作，默认为False。
        get_scale_zp: 是否返回缩放因子和零点，默认为False。
    """
    org_w_shape = w.shape                                                   # 记录原始权重张量的形状，以便后续恢复
    if q_group_size > 0:                                                    # 如果指定了量化组大小
        assert org_w_shape[-1] % q_group_size == 0                          # 则要求最后一个维度可以被量化组大小整除
        w = w.reshape(-1, q_group_size)                                     # 将权重张量重塑为二维张量，形状为[-1, q_group_size]
    assert w.dim() == 2                                                     # 确保权重张量为二维张量

    # 计算缩放因子和零点
    if zero_point:                                                          # 如果使用零点量化
        max_val = w.amax(dim=1, keepdim=True)                               # 沿第一维（每组）计算最大值
        min_val = w.amin(dim=1, keepdim=True)                               # 沿第一维（每组）计算最小值
        max_int = 2**n_bit - 1                                              # 计算量化后的整数最大值
        min_int = 0                                                         # 计算量化后的整数最小值
        scales = (max_val - min_val).clamp(min=1e-5) / max_int              # 计算每组的缩放因子
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)   # 计算每组的零点
    else:  # we actually never used this                                    # 对称量化
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)                         # 沿第一维（每组）计算绝对值的最大值
        max_val = max_val.clamp(min=1e-5)                                   # 限制最大值的下限
        max_int = 2 ** (n_bit - 1) - 1                                      # 计算量化后的整数最大值
        min_int = -(2 ** (n_bit - 1))                                       # 计算量化后的整数最小值
        scales = max_val / max_int                                          # 计算每组的缩放因子
        zeros = 0                                                           # 零点为0

    # 确保缩放因子和零点没有NaN值
    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    # 量化权重张量
    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0                                        # 确保量化后的权重张量没有NaN值

    w = w.reshape(org_w_shape)                                              # 恢复原始形状

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w


@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    w_bit,
    q_config,
):
    """
    对整个模型的权重进行模拟量化（伪量化）。
    它通过遍历模型中的所有可量化线性层，调用`pseudo_quantize_tensor`方法对每个权重张量进行量化处理。

    Args:
        model: 待量化的模型。
        w_bit: 权重量化的比特数。
        q_config: 量化配置参数。
    """
    from .pre_quant import get_blocks, get_named_linears

    layers = get_blocks(model)                                              # 获取模型中的所有可量化模块（如Transformer的解码器层），存储在`layers`中
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):    # 遍历每一个可量化模块
        named_linears = get_named_linears(layers[i])                        # 获取模块中所有命名的线性层，键为层名称，值为对应的线性层模块
        for n, m in named_linears.items():                                  # 遍历每一个线性层
            m.cuda()
            m.weight.data = pseudo_quantize_tensor(                         # 量化权重张量
                m.weight.data, n_bit=w_bit, **q_config
            )
            m.cpu()


@torch.no_grad()
def real_quantize_model_weight(model, w_bit, q_config, init_only=False):
    """
    对模型的权重进行实际量化，将浮点数权重转换为低精度整数表示，并用`WQLinear`模块替换原始的线性层模块。
    该函数支持初始化量化过程，以及完成量化的完整过程。
    这是在模型部署前对模型进行量化的重要步骤，有助于减小模型大小、提高推理速度，并降低内存消耗。

    Args:
        model: 待量化的模型。
        w_bit: 权重量化的比特数。
        q_config: 量化配置参数。
        init_only: 是否仅初始化量化，默认为False。
    """
    from .qmodule import WQLinear
    from .pre_quant import get_blocks, get_named_linears

    assert q_config["zero_point"], "We only support zero_point quantization now."       # 仅支持零点量化

    layers = get_blocks(model)                                              # 获取模型中的所有可量化模块（如Transformer的解码器层），存储在`layers`中
    for i in tqdm(                                                          # 遍历每一个可量化模块
        range(len(layers)),
        desc="real weight quantization..." + ("(init only)" if init_only else ""),
    ):
        layer = layers[i]                                                   # 获取当前模块
        named_linears = get_named_linears(layer)                            # 获取模块中所有命名的线性层，键为层名称，值为对应的线性层模块
        scale_activations(layer)                                            # 将激活函数替换为带有缩放因子的`ScaledActivation`模块

        for name, module in named_linears.items():                          # 遍历每一个线性层
            if init_only:                                                   # 如果仅初始化量化
                q_linear = WQLinear.from_linear(                            # 根据原始线性层创建一个量化线性层`WQLinear`实例
                    module, w_bit, q_config["q_group_size"], True
                )
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)                       # 将模块中的原始线性层替换为量化线性层
            else:                                                           # 否则，进行实际量化
                module.cuda()
                module.weight.data, scales, zeros = pseudo_quantize_tensor( # 量化权重张量，并获取缩放因子和零点
                    module.weight.data, n_bit=w_bit, get_scale_zp=True, **q_config
                )
                # scales = scales.t().contiguous()
                # zeros = zeros.t().contiguous()
                q_linear = WQLinear.from_linear(                            # 根据原始线性层创建一个量化线性层`WQLinear`实例
                    module, w_bit, q_config["q_group_size"], False, scales, zeros
                )
                module.cpu()
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)                       # 将模块中的原始线性层替换为量化线性层
                torch.cuda.empty_cache()
                gc.collect()

    torch.cuda.empty_cache()
    gc.collect()

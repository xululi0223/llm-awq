import gc
import torch
import torch.nn as nn

from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.activations import GELUActivation

from .qmodule import ScaledActivation
from ..utils.module import get_op_by_name, get_op_name, set_op_by_name

__all__ = ["auto_scale_block", "apply_scale"]


@torch.no_grad()
def get_weight_scale(weight, q_group_size=-1):
    """
    计算权重张量的缩放因子。
    通过对权重的绝对值进行归一化处理，并计算每个特征的平均缩放因子，以用于后续的量化操作。

    Args:
        weight: 权重张量。
        q_group_size: 量化组大小，默认为-1，表示不进行分组。
    """
    org_shape = weight.shape                        # 保存原始形状
    if q_group_size > 0:                            # 如果量化组大小大于0，则对权重进行分组
        weight = weight.view(-1, q_group_size)
    scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)   # 计算每个元素的缩放因子，用每个元素的绝对值除以其所在组的最大绝对值
    scale = scale.view(org_shape)                   # 恢复原始形状
    scale = scale.mean(0)                           # 沿第一个维度（通常是输出特征维度）计算缩放因子的平均值
    return scale


@torch.no_grad()
def get_act_scale(x):
    """
    计算激活张量的缩放因子。
    通过对激活的绝对值取平均，得到每个特征维度的缩放因子，以便进行量化或归一化处理。
    """
    return x.abs().view(-1, x.shape[-1]).mean(0)    # 计算激活张量的绝对值，取平均值，得到每个特征维度的缩放因子


@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    """
    对层归一化和相关的全连接层进行缩放调整。
    通过将层归一化的权重和偏置除以缩放因子，同时将全连接层的权重乘以缩放因子，实现激活值的动态调整和量化适配。

    Args:
        ln: 层归一化模块。
        fcs: 全连接层模块列表。
        scales: 缩放因子张量。
    """
    if not isinstance(fcs, list):                   # 如果fcs不是列表，则转换为列表
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    ln.weight.div_(scales)                          # 将层归一化的权重除以缩放因子
    if hasattr(ln, "bias") and ln.bias is not None: # 如果层归一化有偏置，则也除以缩放因子
        ln.bias.div_(scales)

    for fc in fcs:                                  # 调整全连接层权重，乘以缩放因子
        fc.weight.mul_(scales.view(1, -1))

    # 确保调整后的参数没有NaN值
    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    """
    对两个连续的全连接层进行缩放调整。
    通过将第一个全连接层的权重除以缩放因子，同时将第二个全连接层的权重乘以相同的缩放因子，实现激活值的动态调整和量化适配。

    Args:
        fc1: 第一个全连接层模块。
        fc2: 第二个全连接层模块。
        scales: 缩放因子张量。
    """
    assert isinstance(fc1, nn.Linear)               # 确保fc1是全连接层
    assert isinstance(fc2, nn.Linear)               # 确保fc2是全连接层
    # assert fc1.out_features == fc2.in_features

    scales = scales.to(fc1.weight.device)

    # fc1.weight.div_(scales.view(-1, 1))
    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))  # 仅对权重矩阵的最后`scales.size(0)`行进行缩放调整
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))                      # 如果fc1有偏置，则也除以缩放因子

    fc2.weight.mul_(scales.view(1, -1))             # 调整fc2的权重，乘以缩放因子

    # 确保调整后的参数没有NaN值
    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_gelu_fc(gelu, fc, scales):
    """
    用于对GELU激活函数和关联的全连接层进行缩放调整。
    通过将全连接层的权重乘以缩放因子，实现激活值的动态调整和量化适配，同时确保激活函数模块的类型正确。

    Args:
        gelu: GELU激活函数模块。
        fc: 全连接层模块。
        scales: 缩放因子张量。
    """
    assert isinstance(gelu, (nn.GELU, BloomGelu, GELUActivation))   # 确保gelu是GELU激活函数
    assert isinstance(fc, nn.Linear)                                # 确保fc是全连接层

    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))         # 调整fc的权重，乘以缩放因子

    # 确保调整后的参数没有NaN值
    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def auto_scale_block(module, module_kwargs, w_bit, q_config, input_feat):
    """
    自动位特定的模块（如不同类型的Transformer解码器层）计算和搜索最佳的缩放因子，以便在模型量化过程中调整权重和激活值。

    Args:
        module: 当前处理的模块，通常是一个Transformer解码器层。
        module_kwargs: 模块的关键字参数。
        w_bit: 权重量化的位数。
        q_config: 量化配置，用于权重量化函数。
        input_feat: 输入特征字典，包含各层输入的张量。
    """
    from .quantizer import pseudo_quantize_tensor

    # firstly, get the weight quantize function
    if w_bit is not None:                                           # 如果权重位数不为空，调用`pseudo_quantize_tensor`函数

        def w_quantize_func(p):
            return pseudo_quantize_tensor(
                p,
                n_bit=w_bit,
                **q_config,
            ).detach()

    else:                                                           # 否则，直接返回权重张量p

        def w_quantize_func(p):
            return p

    if "use_cache" in module_kwargs:                                # 如果关键字参数中有`use_cache`，则将其移除。避免在缩放过程中使用缓存机制
        module_kwargs.pop("use_cache")

    # find the best scale ratio
    def _search_module_scale(block, linears2scale: list, x, kwargs={}):
        """
        通过遍历不同的缩放比例，搜索出导致最小输出误差的最佳缩放因子。

        Args:
            block: 当前量化的模块块。
            linears2scale: 需要调整权重的的全连接层列表。
            x: 输入张量。
            kwargs: 其他关键字参数。
        """
        # w: co, ci
        # x: n, ci
        x = x.to(next(block.parameters()).device)
        # 计算原始输出
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        x_max = get_act_scale(x)                                    # 计算输入张量`x`的激活缩放因子

        best_error = float("inf")                                   # 初始化最小误差为无穷大
        best_ratio = -1                                             # 初始化最佳缩放比例为-1
        best_scales = None                                          # 初始化最佳缩放因子为None

        n_grid = 20                                                 # 定义搜索网格大小
        history = []                                                # 初始化损失历史记录

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}    # 保存原始模块状态字典，以便在每次迭代后恢复
        for ratio in range(n_grid):                                 # 遍历不同比率
            ratio = ratio * 1 / n_grid                              # 生成0~1之间的比率
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)      # 根据比率调整缩放因子，并确保最小值为1e-4
            scales = scales / (scales.max() * scales.min()).sqrt()  # 对缩放因子进行归一化处理，以平衡最大和最小值
            for fc in linears2scale:                                # 遍历每个需要缩放的全连接层
                fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))     # 调整全连接层的权重，乘以缩放因子
                fc.weight.data = w_quantize_func(fc.weight.data) / (scales.view(1, -1))     # 对调整后的权重进行量化处理，并除以缩放因子
            out = block(x, **kwargs)                                # 计算量化后模块的输出
            if isinstance(out, tuple):
                out = out[0]

            loss = (
                (org_out - out).float().pow(2).mean().item()        # 计算输出误差，即原始输出与量化后输出的均方误差
            )  # float prevents overflow
            history.append(loss)                                    # 记录误差历史
            is_best = loss < best_error                             # 判断当前误差是否小于最小误差
            if is_best:                                             # 如果是最佳误差，则更新最佳误差、最佳比率和最佳缩放因子
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            block.load_state_dict(org_sd)                           # 恢复原始模块状态字典
        if best_ratio == -1:
            print(history)
            raise Exception
        # print(best_ratio)
        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales     # 确保最佳缩放因子没有NaN值
        return best_scales.detach()

    def _auto_get_scale(prev_op, layers, inp, module2inspect=None, kwargs={}):
        """
        调用`_search_module_scale`函数，搜索模块的最佳缩放因子。

        Args:
            prev_op: 前一个操作模块。
            layers: 需要调整权重的全连接层列表。
            inp: 输入张量。
            module2inspect: 需要检查输出差异的模块，如果给定，则使用该模块；否则，默认使用`layers[0]`。
            kwargs: 其他关键字参数。
        """
        # module2inspect: if given, we will check the output diff of this module instead of layers
        if module2inspect is None:                                  # 如果未指定`module2inspect`，则断言`layers`长度为1，并使用`layers[0]`作为检查模块
            assert len(layers) == 1
            module2inspect = layers[0]

        scales = _search_module_scale(module2inspect, layers, inp, kwargs)      # 调用`_search_module_scale`函数，搜索模块的最佳缩放因子
        scales = scales.detach().cpu()
        # prev_op_name, [layer_name], scale
        return (
            get_op_name(module, prev_op),                           # 前一个操作的名称
            tuple([get_op_name(module, m) for m in layers]),        # 当前层列表的名称
            scales,                                                 # 缩放因子
        )

    scales_list = []  # return the searched scales 初始化缩放因子列表

    # 处理`OPTDecoderLayer`模块
    if isinstance(module, OPTDecoderLayer):
        # attention input
        scales_list.append(
            _auto_get_scale(
                prev_op=module.self_attn_layer_norm,                # 前一个操作为`module.self_attn_layer_norm`
                layers=[                                            # 需要调整的全连接层为`q_proj`, `k_proj`, `v_proj`
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],                 # 输入张量为`input_feat["self_attn.q_proj"]`
                module2inspect=module.self_attn,                    # 需要检查输出差异的模块为`module.self_attn`
                kwargs=module_kwargs,
            )
        )
        # attn out
        scales_list.append(
            _auto_get_scale(
                prev_op=module.self_attn.v_proj,                    # 前一个操作为`module.self_attn.v_proj`
                layers=[module.self_attn.out_proj],                 # 需要调整的全连接层为`out_proj`
                inp=input_feat["self_attn.out_proj"],               # 输入张量为`input_feat["self_attn.out_proj"]`
            )
        )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.final_layer_norm,                    # 前一个操作为`module.final_layer_norm`
                layers=[module.fc1],                                # 需要调整的全连接层为`fc1`
                inp=input_feat["fc1"],                              # 输入张量为`input_feat["fc1"]`
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.fc1,                                 # 前一个操作为`module.fc1`
                layers=[module.fc2],                                # 需要调整的全连接层为`fc2`
                inp=input_feat["fc2"],                              # 输入张量为`input_feat["fc2"]`
            )
        )

    # 处理`LlamaDecoderLayer`模块
    elif isinstance(module, LlamaDecoderLayer):
        # attention input
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,                     # 前一个操作为`module.input_layernorm`
                layers=[                                            # 需要调整的全连接层为`q_proj`, `k_proj`, `v_proj`
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],                 # 输入张量为`input_feat["self_attn.q_proj"]`
                module2inspect=module.self_attn,                    # 需要检查输出差异的模块为`module.self_attn`
                kwargs=module_kwargs,
            )
        )
        # attn out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:    # 如果`v_proj`和`o_proj`的权重形状相同，则对attn out进行缩放调整
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.self_attn.v_proj,                # 前一个操作为`module.self_attn.v_proj`
                    layers=[module.self_attn.o_proj],               # 需要调整的全连接层为`o_proj`
                    inp=input_feat["self_attn.o_proj"],             # 输入张量为`input_feat["self_attn.o_proj"]`
                )
            )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,            # 前一个操作为`module.post_attention_layernorm`
                layers=[module.mlp.gate_proj, module.mlp.up_proj],  # 需要调整的全连接层为`gate_proj`, `up_proj`
                inp=input_feat["mlp.gate_proj"],                    # 输入张量为`input_feat["mlp.gate_proj"]`
                module2inspect=module.mlp,                          # 需要检查输出差异的模块为`module.mlp`
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.up_proj,                         # 前一个操作为`module.mlp.up_proj`
                layers=[module.mlp.down_proj],                      # 需要调整的全连接层为`down_proj`
                inp=input_feat["mlp.down_proj"],                    # 输入张量为`input_feat["mlp.down_proj"]`
            )
        )

    # 处理`BloomBlock`模块
    elif isinstance(module, BloomBlock):
        # attention input
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,                     # 前一个操作为`module.input_layernorm`
                layers=[module.self_attention.query_key_value],     # 需要调整的全连接层为`self_attention.query_key_value`
                inp=input_feat["self_attention.query_key_value"],   # 输入张量为`input_feat["self_attention.query_key_value"]`
                module2inspect=module,                              # 需要检查输出差异的模块为`module`
                kwargs=module_kwargs,
            )
        )
        # attn out
        # Please refer to https://github.com/mit-han-lab/llm-awq/issues/2#issuecomment-1606297469
        """
        scales_list.append(_auto_get_scale(
            prev_op=module.self_attention.query_key_value,
            layers=[module.self_attention.dense],
            inp=input_feat['self_attention.dense'],
        ))
        """
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,            # 前一个操作为`module.post_attention_layernorm`
                layers=[module.mlp.dense_h_to_4h],                  # 需要调整的全连接层为`mlp.dense_h_to_4h`
                inp=input_feat["mlp.dense_h_to_4h"],                # 输入张量为`input_feat["mlp.dense_h_to_4h"]`
                module2inspect=module,                              # 需要检查输出差异的模块为`module`
                kwargs=module_kwargs,
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.gelu_impl,                       # 前一个操作为`module.mlp.gelu_impl`
                layers=[module.mlp.dense_4h_to_h],                  # 需要调整的全连接层为`mlp.dense_4h_to_h`
                inp=input_feat["mlp.dense_4h_to_h"],                # 输入张量为`input_feat["mlp.dense_4h_to_h"]`
            )
        )

    # 处理包含`mpt`字符串的模块
    elif "mpt" in str(module.__class__).lower():
        # attention input
        scales_list.append(
            _auto_get_scale(
                prev_op=module.norm_1,                              # 前一个操作为`module.norm_1`
                layers=[module.attn.Wqkv],                          # 需要调整的全连接层为`attn.Wqkv`
                inp=input_feat["attn.Wqkv"],                        # 输入张量为`input_feat["attn.Wqkv"]`
                module2inspect=module.attn,                         # 需要检查输出差异的模块为`module.attn`
                kwargs=module_kwargs,
            )
        )

        # attn out
        scales_list.append(
            _auto_get_scale(
                prev_op=module.attn.Wqkv,                           # 前一个操作为`module.attn.Wqkv`
                layers=[module.attn.out_proj],                      # 需要调整的全连接层为`attn.out_proj`
                inp=input_feat["attn.out_proj"],                    # 输入张量为`input_feat["attn.out_proj"]`
            )
        )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.norm_2,                              # 前一个操作为`module.norm_2`
                layers=[module.ffn.up_proj],                        # 需要调整的全连接层为`ffn.up_proj`
                inp=input_feat["ffn.up_proj"],                      # 输入张量为`input_feat["ffn.up_proj"]`
                module2inspect=module.ffn,                          # 需要检查输出差异的模块为`module.ffn`
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ffn.act,                             # 前一个操作为`module.ffn.act`
                layers=[module.ffn.down_proj],                      # 需要调整的全连接层为`ffn.down_proj`
                inp=input_feat["ffn.down_proj"],                    # 输入张量为`input_feat["ffn.down_proj"]`
            )
        )

    # 处理包含`falcon`字符串的模块
    elif "falcon" in str(module.__class__).lower():
        # attn out
        # Haotian: TBD: need to handle repeated scales for MQ
        """
        scales_list.append(_auto_get_scale(
            prev_op=module.self_attention.query_key_value,
            layers=[module.self_attention.dense],
            inp=input_feat['self_attention.dense'],
        ))
        """
        # fc1, as long as it is scaled, everything is screwed up
        if "falcon-7b" in str(module.__class__).lower():            # 如果模块包含`falcon-7b`字符串
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.input_layernorm,                 # 前一个操作为`module.input_layernorm`
                    layers=[                                        # 需要调整的全连接层为`mlp.dense_h_to_4h`, `self_attention.query_key_value`
                        module.mlp.dense_h_to_4h,
                        module.self_attention.query_key_value,
                    ],
                    inp=input_feat["self_attention.query_key_value"],   # 输入张量为`input_feat["self_attention.query_key_value"]`
                    module2inspect=module,                          # 需要检查输出差异的模块为`module`
                    kwargs=module_kwargs,
                )
            )
        elif "falcon-40b" in str(module.__class__).lower():         # 如果模块包含`falcon-40b`字符串
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.ln_attn,                         # 前一个操作为`module.ln_attn`
                    layers=[module.self_attention.query_key_value], # 需要调整的全连接层为`self_attention.query_key_value`
                    inp=input_feat["self_attention.query_key_value"],   # 输入张量为`input_feat["self_attention.query_key_value"]`
                    module2inspect=module,                          # 需要检查输出差异的模块为`module`
                    kwargs=module_kwargs,
                )
            )
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.ln_mlp,                          # 前一个操作为`module.ln_mlp`
                    layers=[module.mlp.dense_h_to_4h],              # 需要调整的全连接层为`mlp.dense_h_to_4h`
                    inp=input_feat["mlp.dense_h_to_4h"],            # 输入张量为`input_feat["mlp.dense_h_to_4h"]`
                    module2inspect=module,                          # 需要检查输出差异的模块为`module`
                    kwargs=module_kwargs,
                )
            )
        else:
            raise NotImplementedError(
                "Unknown Falcon architecture, currently only falcon-7b and falcon-40b are supported"
            )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.act,                             # 前一个操作为`module.mlp.act`
                layers=[module.mlp.dense_4h_to_h],                  # 需要调整的全连接层为`mlp.dense_4h_to_h`
                inp=input_feat["mlp.dense_4h_to_h"],                # 输入张量为`input_feat["mlp.dense_4h_to_h"]`
            )
        )
    
    # 处理包含`bigcode`字符串的模块
    elif "bigcode" in str(module.__class__).lower():
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ln_1,                                # 前一个操作为`module.ln_1`
                layers=[module.attn.c_attn],                        # 需要调整的全连接层为`attn.c_attn`
                inp=input_feat["attn.c_attn"],                      # 输入张量为`input_feat["attn.c_attn"]`
                module2inspect=module.attn,                         # 需要检查输出差异的模块为`module.attn`
                kwargs=module_kwargs,
            )
        )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ln_2,                                # 前一个操作为`module.ln_2`
                layers=[module.mlp.c_fc],                           # 需要调整的全连接层为`mlp.c_fc`
                inp=input_feat["mlp.c_fc"],                         # 输入张量为`input_feat["mlp.c_fc"]`
                module2inspect=module.mlp,                          # 需要检查输出差异的模块为`module.mlp`
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.act,                             # 前一个操作为`module.mlp.act`
                layers=[module.mlp.c_proj],                         # 需要调整的全连接层为`mlp.c_proj`
                inp=input_feat["mlp.c_proj"],                       # 输入张量为`input_feat["mlp.c_proj"]`
            )
        )

    # 处理包含`neox`字符串的模块
    elif "neox" in str(module.__class__).lower():
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,                     # 前一个操作为`module.input_layernorm`
                layers=[module.attention.query_key_value],          # 需要调整的全连接层为`attention.query_key_value`
                inp=input_feat["attention.query_key_value"],        # 输入张量为`input_feat["attention.query_key_value"]`
                module2inspect=module.attention,                    # 需要检查输出差异的模块为`module.attention`
                kwargs=module_kwargs,
            )
        )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,            # 前一个操作为`module.post_attention_layernorm`
                layers=[module.mlp.dense_h_to_4h],                  # 需要调整的全连接层为`mlp.dense_h_to_4h`
                inp=input_feat["mlp.dense_h_to_4h"],                # 输入张量为`input_feat["mlp.dense_h_to_4h"]`
                module2inspect=module.mlp,                          # 需要检查输出差异的模块为`module.mlp`
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.act,                             # 前一个操作为`module.mlp.act`
                layers=[module.mlp.dense_4h_to_h],                  # 需要调整的全连接层为`mlp.dense_4h_to_h`
                inp=input_feat["mlp.dense_4h_to_h"],                # 输入张量为`input_feat["mlp.dense_4h_to_h"]`
            )
        )
    else:
        raise NotImplementedError(f"{type(module)} not supported yet!")

    return scales_list


def apply_scale(module, scales_list, input_feat_dict=None):
    """
    将搜索到的缩放因子应用到对应的模块和层。
    它根据前一个操作模块的类型调用相应的缩放函数，并可选地对输入特征进行缩放调整。

    Args:
        module: 当前处理的模块，通常是一个Transformer解码器层。
        scales_list: 缩放因子列表，包含前一个操作名称、需要调整的层名称和缩放因子。
        input_feat_dict: 输入特征字典，用于对输入张量进行缩放调整。
    """
    for prev_op_name, layer_names, scales in scales_list:                   # 遍历缩放因子列表，获取前一个操作名称、需要调整的层名称和缩放因子
        prev_op = get_op_by_name(module, prev_op_name)                      # 获取前一个操作模块
        layers = [get_op_by_name(module, name) for name in layer_names]     # 获取需要调整的层模块列表

        prev_op.cuda()
        for layer in layers:
            layer.cuda()
        scales.cuda()

        # 根据前一个操作模块的类型调用相应的缩放函数
        if isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales)                         # 对两个连续的全连接层进行缩放调整
        elif isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm)):
            scale_ln_fcs(prev_op, layers, scales)                           # 对层归一化和相关的全连接层进行缩放调整
        elif isinstance(prev_op, (nn.GELU, BloomGelu, GELUActivation)):
            new_module = ScaledActivation(prev_op, scales)                  # 创建一个新的激活函数模块，用于对GELU激活函数进行缩放调整
            set_op_by_name(module, prev_op_name, new_module)                # 更新前一个操作模块
            scale_gelu_fc(prev_op, layers[0], scales)                       # 对GELU激活函数和关联的全连接层进行缩放调整
        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

        # apply the scaling to input feat if given; prepare it for clipping
        if input_feat_dict is not None:                                     # 如果提供了输入特征字典，则对每个需要调整的层的输入张量进行缩放调整
            for layer_name in layer_names:                                  # 遍历需要调整的层名称
                inp = input_feat_dict[layer_name]                           # 获取输入张量
                inp.div_(scales.view(1, -1).to(inp.device))                 # 对输入张量进行缩放调整

        prev_op.cpu()
        for layer in layers:
            layer.cpu()
        scales.cpu()

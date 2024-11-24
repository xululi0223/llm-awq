import torch
import torch.nn as nn
import tqdm
import gc
import functools
from collections import defaultdict
from typing import List

from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from tinychat.models import LlavaLlamaForCausalLM

from .auto_scale import auto_scale_block, apply_scale
from .auto_clip import auto_clip_block, apply_clip

__all__ = ["run_awq"]


def get_named_linears(module):
    """
    从给定的模块中筛选出所有的线性层，并返回一个字典，键为线性层的名字，值为线性层对象。
    这在后续的量化和裁剪过程中，可以方便地访问和操作这些线性层。

    Args:
        module: torch.nn.Module, 给定的模块。
    """
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    """
    获取不同类型模型中的关键层次结构（通常是Transformer的层）。
    根据模型的类型或名称，选择相应的属性路径来提取这些层。
    这对于统一处理不同模型架构的量化和裁剪操作非常有用。

    Args:
        model: torch.nn.Module, 要提取层的模型实例。
    """
    if model.__class__.__name__ == "LlamaForCausalLM":      # 处理`LlamaForCausalLM`模型
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":   # 处理`LlavaLlamaForCausalLM`模型
        # layers = [model.model.layers, model.model.vision_tower.vision_tower.vision_model.encoder.layers]
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):                 # 处理`OPTForCausalLM`模型
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):               # 处理`BloomForCausalLM`模型
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():             # 处理`mpt`模型
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():          # 处理`falcon`模型
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():         # 处理`bigcode`模型
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():            # 处理`neox`模型
        layers = model.gpt_neox.layers
    else:
        raise NotImplementedError(type(model))
    return layers


def move_embed(model, device):
    """
    将模型中的嵌入层（如词嵌入和位置嵌入）移动到指定的设备上。
    不同模型架构的嵌入曾位于不同的属性路径，因此需要根据模型类型进行相应的处理。

    Args:
        model: torch.nn.Module, 要移动嵌入层的模型实例。
        device: str, 目标设备名称，如`cpu`或`cuda:0`。
    """
    if isinstance(model, LlamaForCausalLM):                 # 处理`LlamaForCausalLM`模型
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif isinstance(model, LlavaLlamaForCausalLM):          # 处理`LlavaLlamaForCausalLM`模型
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.vision_tower.vision_tower.vision_model.embeddings.to(device)
    elif isinstance(model, OPTForCausalLM):                 # 处理`OPTForCausalLM`模型
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            device
        )
    elif isinstance(model, BloomForCausalLM):               # 处理`BloomForCausalLM`模型
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = (
            model.transformer.word_embeddings_layernorm.to(device)
        )
    elif "mpt" in str(model.__class__).lower():             # 处理`mpt`模型
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():          # 处理`falcon`模型
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    elif "bigcode" in str(model.__class__).lower():         # 处理`bigcode`模型
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.wpe = model.transformer.wpe.to(device)
        model.transformer.drop = model.transformer.drop.to(device)
    elif "neox" in str(model.__class__).lower():            # 处理`neox`模型
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(device)
        model.gpt_neox.emb_dropout = model.gpt_neox.emb_dropout.to(device)
        model.embed_out = model.embed_out.to(device)
    else:
        raise NotImplementedError(type(model))


@torch.no_grad()
def run_awq(
    model,
    enc,
    w_bit,
    q_config,
    n_samples=512,
    seqlen=512,
    auto_scale=True,
    mse_range=True,
    # some configs for ablation study
    calib_data="pileval",
):
    """
    执行自适应权重量化（AWQ）的主要流程。
    该过程包括获取校准数据、捕获输入特征、自动缩放和裁剪权重，以最小化量化误差并保持模型性能。

    Args:
        model: torch.nn.Module, 待量化的模型实例。
        enc: transformers.PreTrainedTokenizer, 编码器，用于处理校准数据。
        w_bit: int, 权重量化的位数。
        q_config: dict, 量化配置参数，通常包括`q_group_size`等。
        n_samples: int, 校准样本数量，默认512。
        seqlen: int, 序列长度，默认512。
        auto_scale: bool, 是否执行自动缩放，默认为True。
        mse_range: bool, 是否执行均方误差裁剪，默认为True。
        calib_data: str, 校准数据来源，默认为`pileval`。
    """
    from ..utils.calib_data import get_calib_dataset
    from ..utils.module import append_str_prefix, get_op_name

    if "bigcode" in str(model.__class__).lower():                               # 处理`bigcode`模型
        # otherwise attention_mask will always be on cpu.
        model.transformer.bias = model.transformer.bias.to("cuda")

    layers = get_blocks(model)                                                  # 提取模型中的关键层列表

    samples = get_calib_dataset(                                                # 获取校准数据集
        data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen
    )
    samples = torch.cat(samples, dim=0)                                         # 将多个样本沿第0维拼接成一个大的张量

    inps = []                                                                   # 存储捕获的输入张量
    layer_kwargs = {}                                                           # 存储捕获的关键字参数

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")                                                   # 将嵌入层移动到GPU上

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    # 定义一个Catcher类，用于捕获输入和关键字参数
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)                                                    # 捕获输入
            layer_kwargs.update(kwargs)                                         # 捕获关键字参数
            raise ValueError  # early exit to break later inference             # 提前退出以中断后续推理

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])                                              # 使用Catcher类包装第一个层，以捕获输入和关键字参数
    try:
        model(samples.to(next(model.parameters()).device))                      # 运行模型，捕获输入和关键字参数
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore                                     # 恢复原始的第一个层
    inps = inps[0]                                                              # 获取捕获的输入张量

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")                                                    # 将嵌入层移动到CPU上

    gc.collect()
    torch.cuda.empty_cache()

    awq_results = {                                                             # 存储缩放和裁剪的结果
        "scale": [],
        "clip": [],
    }

    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ..."):              # 遍历所有层
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)                                # 获取当前层中的所有线性层

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            """
            用于捕获线性层的输入特征。

            Args:
                m: 被钩子附加的模块。
                x: 输入张量的元组。
                y: 输出张量的元组。
                name: 线性层的名字。
                feat_dict: 存储输入特征的字典。
            """
            x = x[0]                                                            # 获取输入张量
            x = x.detach().cpu()
            feat_dict[name].append(x)                                           # 将处理后的输入张量添加到字典中

        input_feat = defaultdict(list)                                          # 用于存储每个线性层的输入特征
        handles = []                                                            # 用于存储钩子句柄
        for name in named_linears:                                              # 遍历所有线性层
            handles.append(
                named_linears[name].register_forward_hook(                      # 对每个线性层，注册一个前向钩子，用于捕获输入特征
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]                                   # 执行前向传播，获取输出作为下一层的输入
        for h in handles:                                                       # 移除所有钩子
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}    # 将输入特征拼接成一个大的张量

        # Clear GPU memory
        torch.cuda.empty_cache()

        if (
            auto_scale                                                          # 如果需要自动缩放
        ):  # if it applies, we should also modify the input_feat with scales
            scales_list = auto_scale_block(                                     # 针对当前层执行自动缩放
                layer,                                                          # 当前层
                layer_kwargs,                                                   # 关键字参数
                w_bit=w_bit,                                                    # 权重位数
                q_config=q_config,                                              # 量化配置参数
                input_feat=input_feat,                                          # 捕获到的输入特征
            )
            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat)     # 将缩放结果应用到当前层
            # append prefix to make names global
            awq_results["scale"] += append_str_prefix(                          # 将缩放结果添加到`awq_results`中
                scales_list, get_op_name(model, layer) + "."
            )

        # Clear GPU memory
        torch.cuda.empty_cache()

        if mse_range:                                                           # 如果需要均方误差裁剪
            clip_list = auto_clip_block(                                        # 针对当前层执行自动裁剪
                layer,                                                          # 当前层
                w_bit=w_bit,                                                    # 权重位数
                q_config=q_config,                                              # 量化配置参数
                input_feat=input_feat,                                          # 捕获到的输入特征
            )
            apply_clip(layer, clip_list)                                        # 将裁剪结果应用到当前层
            # append prefix to make names global
            awq_results["clip"] += append_str_prefix(                           # 将裁剪结果添加到`awq_results`中
                clip_list, get_op_name(model, layer) + "."
            )

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    return awq_results


def apply_awq(model, awq_results):
    """
    将自适应权重量化过程中计算得到的缩放和裁剪参数应用到整个模型中。
    具体来说，它调用`apply_scale`和`apply_clip`函数，分别应用缩放和裁剪操作。

    Args:
        model: 待量化的模型实例。
        awq_results: 自适应权重量化过程中计算得到的缩放和裁剪参数。
    """
    apply_scale(model, awq_results["scale"])                                    # 应用缩放操作
    apply_clip(model, awq_results["clip"])                                      # 应用裁剪操作

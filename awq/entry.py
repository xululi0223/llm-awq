from lm_eval import evaluator, tasks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse
import os
import json
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model,
    load_checkpoint_in_model,
)
from accelerate.utils.modeling import get_balanced_memory
from awq.utils.parallel import auto_parallel
from awq.quantize.pre_quant import run_awq, apply_awq
from awq.quantize.quantizer import (
    pseudo_quantize_model_weight,
    real_quantize_model_weight,
)
from awq.utils.lm_eval_adaptor import LMEvalAdaptor
from awq.utils.utils import simple_dispatch_model
from datasets import load_dataset
from torch import nn
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path of the hf model")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--tasks", default=None, type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument("--num_fewshot", type=int, default=0)
# model config
parser.add_argument("--parallel", action="store_true", help="enable model parallelism")
# max memory to offload larger models to CPU
parser.add_argument(
    "--max_memory",
    type=str,
    nargs="*",
    help="List of device_id:max_memory pairs to be parsed into a dictionary; "
    + "Example: 0:10GiB 1:10GiB cpu:30GiB; "
    + "mode details here: "
    + "https://huggingface.co/docs/accelerate/usage_guides/big_modeling",
)
parser.add_argument(
    "--auto_parallel",
    action="store_true",
    help="automatically set parallel and batch_size",
)
# quantization config
parser.add_argument("--w_bit", type=int, default=None)
parser.add_argument("--q_group_size", type=int, default=-1)
parser.add_argument("--no_zero_point", action="store_true", help="disable zero_point")
parser.add_argument("--q_backend", type=str, default="fake", choices=["fake", "real"])
# save/load real quantized weights
parser.add_argument("--dump_quant", type=str, default=None, help="save quantized model")
parser.add_argument("--dump_fake", type=str, default=None, help="save fake-quantized model")
parser.add_argument("--load_quant", type=str, default=None, help="load quantized model")
# apply/save/load awq
parser.add_argument("--run_awq", action="store_true", help="perform awq search process")
parser.add_argument(
    "--dump_awq", type=str, default=None, help="save the awq search results"
)
parser.add_argument(
    "--load_awq", type=str, default=None, help="load the awq search results"
)
parser.add_argument(
    "--vila-15",
    action="store_true",
    help="quantizing vila 1.5",
)
args = parser.parse_args()
vila_10_quant_mode = ("llava" in args.model_path.lower() or "vila" in args.model_path.lower()) and not args.vila_15

max_memory = [v.split(":") for v in (args.max_memory or [])]            # 解析`max_memory`参数，将其转换为字典
max_memory = {(int(k) if k.isdigit() else k): v for k, v in max_memory}

if args.auto_parallel:
    gpu_list = auto_parallel(args)                                      # 自动设置并行和批大小

# get quantization config (apart from w_bit)
q_config = {                                                            # 构建量化配置字典
    "zero_point": not args.no_zero_point,  # by default True
    "q_group_size": args.q_group_size,  # whether to use group quantization
}
print("Quantization config:", q_config)

# build model and tokenizer


def build_model_and_enc(model_path):
    """
    根据提供的模型路径加载预训练语言模型和相应的分词器（Tokenizer）。
    函数首先检查模型路径是否存在，然后根据模型路径加载模型的配置信息。

    Args:
        model_path: 预训练模型所在的路径。
    """
    if not os.path.exists(model_path):  # look into ssd
        raise FileNotFoundError(f"{model_path} not found!")
    print(f"* Building model {model_path}")

    # all hf model
    # 处理`vila-1.0`模型
    if vila_10_quant_mode:
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        enc, model, image_processor, context_len = load_pretrained_model(       # 加载预训练模型
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            device="cpu",
            **{"use_cache": False}
        )
    # 处理通用模型
    else:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True) # 加载模型配置
        # Note (Haotian): To avoid OOM after huggingface transformers 4.36.2
        config.use_cache = False
        if "mpt" in config.__class__.__name__.lower():                          # 处理`mpt`模型
            enc = AutoTokenizer.from_pretrained(                                # 加载分词器
                config.tokenizer_name, trust_remote_code=True
            )
        else:                                                                   # 处理其他模型
            enc = AutoTokenizer.from_pretrained(                                # 加载分词器
                model_path, use_fast=False, trust_remote_code=True
            )

    if args.load_quant:  # directly load quantized weights                      # 如果指定了加载预先量化的权重
        print("Loading pre-computed quantized weights...")
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(                           # 根据配置创建模型实例
                config=config, torch_dtype=torch.float16, trust_remote_code=True
            )
        real_quantize_model_weight(                                             # 对模型权重进行真实量化
            model, w_bit=args.w_bit, q_config=q_config, init_only=True
        )

        model.tie_weights()                                                     # 将模型的权重绑定在一起，常用于共享权重参数，如在解码器和嵌入层之间共享权重参数

        # Infer device map
        kwargs = {"max_memory": max_memory} if len(max_memory) else {}          # 根据用户提供的`max_memory`参数构建参数字典
        device_map = infer_auto_device_map(                                     # 根据模型结构和设备内存限制，推断模型各部分应分配到的设备
            model,
            no_split_module_classes=[                                           # 指定哪些模块类型不应被拆分到不同设备上
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
            ],
            **kwargs,
        )
        # Load checkpoint in the model
        load_checkpoint_in_model(                                               # 将预先量化的权重加载到模型中
            model,
            checkpoint=args.load_quant,
            device_map=device_map,
            offload_state_dict=True,
        )
        # Dispatch model
        model = simple_dispatch_model(model, device_map=device_map)             # 将模型分发到不同的设备上

        model.eval()
    else:  # fp16 to quantized                                                  # 如果没有指定加载预先量化的权重
        args.run_awq &= not args.load_awq  # if load_awq, no need to run awq    # 如果指定了加载 AWQ 结果，则不需要运行 AWQ
        # Init model on CPU:
        kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}      # 构建参数字典
        if not vila_10_quant_mode:                                              # 如果不是`vila-1.0`模型
            model = AutoModelForCausalLM.from_pretrained(                       # 加载预训练语言模型
                model_path, config=config, trust_remote_code=True, **kwargs
            )

        model.eval()

        if args.run_awq:                                                        # 执行自适应权重量化
            assert args.dump_awq, "Please save the awq results with --dump_awq" # 确保指定了保存 AWQ 结果的路径

            awq_results = run_awq(                                              # 执行 AWQ
                model,
                enc,
                w_bit=args.w_bit,
                q_config=q_config,
                n_samples=128,
                seqlen=512,
            )
            if args.dump_awq:
                dirpath = os.path.dirname(args.dump_awq)                        # 获取保存 AWQ 结果的目录
                os.makedirs(dirpath, exist_ok=True)                             # 创建目录

                torch.save(awq_results, args.dump_awq)                          # 保存 AWQ 结果
                print("AWQ results saved at", args.dump_awq)

            exit(0)

        if args.load_awq:                                                       # 如果指定了加载 AWQ 结果
            print("Loading pre-computed AWQ results from", args.load_awq)
            awq_results = torch.load(args.load_awq, map_location="cpu")         # 从指定路径加载 AWQ 结果
            apply_awq(model, awq_results)                                       # 将加载的 AWQ 结果应用到模型中

        # weight quantization
        if args.w_bit is not None:                                              # 如果指定了权重位宽
            if args.q_backend == "fake":                                        # 伪量化
                assert (
                    args.dump_quant is None                                     # 确保没有指定保存真实量化的权重
                ), "Need to use real quantization to dump quantized weights"
                pseudo_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)    # 执行伪量化
                if args.dump_fake:                                              # 如果指定了保存伪量化的权重
                    model.save_pretrained(args.dump_fake)                       # 保存伪量化的权重
                    print("Pseudo-quantized models saved at", args.dump_fake)
            elif args.q_backend == "real":  # real quantization                 # 真实量化
                real_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)  # 执行真实量化
                if args.dump_quant:                                             # 如果指定了保存真实量化的权重
                    if not args.dump_quant.endswith("v2.pt"):                   # 如果文件名不以`v2.pt`结尾
                        print("[Info] Auto-change the dump_quant file name to *v2.pt")
                        args.dump_quant = args.dump_quant.replace(".pt", "-v2.pt")  # 修改文件名
                    dirpath = os.path.dirname(args.dump_quant)                  # 获取保存真实量化的权重的目录
                    os.makedirs(dirpath, exist_ok=True)                         # 创建目录

                    print(f"Saving the quantized model at {args.dump_quant}...")
                    torch.save(model.cpu().state_dict(), args.dump_quant)       # 保存真实量化的权重
                    exit(0)
            else:
                raise NotImplementedError

        # Move the model to GPU (as much as possible) for LM evaluation
        kwargs = {
            "max_memory": get_balanced_memory(                                  # 获取平衡的内存
                model, max_memory if len(max_memory) > 0 else None
            )
        }
        device_map = infer_auto_device_map(                                     # 推断模型各部分应分配到的设备
            model,
            # TODO: can we remove this?
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
            ],
            **kwargs,
        )
        model = dispatch_model(model, device_map=device_map)                    # 将模型分发到不同的设备上

    return model, enc


def main():
    if args.output_path is not None and os.path.exists(args.output_path):       # 检查和处理输出路径
        # print(f"Results {args.output_path} already generated. Exit.")
        print(f"Results {args.output_path} already generated. Overwrite.")
        # exit()

    if args.dump_awq and os.path.exists(args.dump_awq):                         # 检查和处理 AWQ 结果的保存路径
        print(f"Found existing AWQ results {args.dump_awq}, exit.")
        exit()

    # a hack here to auto set model group
    model, enc = build_model_and_enc(args.model_path)                           # 构建模型和分词器

    if args.tasks is not None:
        # https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L206
        if args.tasks == "wikitext":                                            # 处理`wikitext`任务
            testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")   # 加载数据集
            testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")    # 将加载的测试集文本连接成一个大的字符串，使用分词器对其进行编码
            model.seqlen = 2048                                                 # 设置模型的序列长度
            testenc = testenc.input_ids.to(model.device)                        # 提取输入 ID
            nsamples = testenc.numel() // model.seqlen                          # 计算样本数量
            model = model.eval()
            nlls = []                                                           # 用于存储每个批次的负对数似然
            for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):          # 遍历每个批次进行评估
                batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(   # 获取当前批次的输入 ID
                    model.device
                )
                with torch.no_grad():
                    lm_logits = model(batch).logits                             # 获取模型的输出logits
                shift_logits = lm_logits[:, :-1, :].contiguous().float()        # 去除logits的最后一个时间步
                shift_labels = testenc[
                    :, (i * model.seqlen) : ((i + 1) * model.seqlen)            # 去除标签的第一个时间步
                ][:, 1:]
                loss_fct = nn.CrossEntropyLoss()                                # 实例化交叉熵损失函数
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)     # 计算交叉熵损失
                )
                neg_log_likelihood = loss.float() * model.seqlen                # 计算负对数似然
                nlls.append(neg_log_likelihood)                                 # 将负对数似然添加到列表中

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))    # 计算困惑度
            print(ppl.item())

            results = {"ppl": ppl.item()}
            if args.output_path is not None:
                os.makedirs(os.path.dirname(args.output_path), exist_ok=True)   # 创建输出目录
                with open(args.output_path, "w") as f:
                    json.dump(results, f, indent=2)                             # 保存结果
        else:
            task_names = args.tasks.split(",")                                  # 分割任务名称，获取多个任务名称列表

            lm_eval_model = LMEvalAdaptor(args.model_path, model, enc, args.batch_size)     # 创建评估适配器，将模型、分词器和批大小传入
            results = evaluator.simple_evaluate(                                # 执行评估
                model=lm_eval_model,
                tasks=task_names,
                batch_size=args.batch_size,
                no_cache=True,
                num_fewshot=args.num_fewshot,
            )

            print(evaluator.make_table(results))                                # 打印评估结果

        if args.output_path is not None:                                        # 如果指定了输出路径
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)       # 创建输出目录
            # otherwise cannot save
            results["config"]["model"] = args.model_path                        # 将模型路径添加到结果中
            with open(args.output_path, "w") as f:
                json.dump(results, f, indent=2)                                 # 保存结果


if __name__ == "__main__":
    main()

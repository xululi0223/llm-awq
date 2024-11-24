import os
import torch
import gc


def auto_parallel(args):
    """
    根据模型的大小自动配置并行计算资源（GPU）的数量，并相应地设置环境变量`CUDA_VISIBLE_DEVICES`来限制程序可见的GPU设备。
    """
    model_size = args.model_path.split("-")[-1]         # 提取模型大小的单位
    if model_size.endswith("m"):                        # 如果模型大小的单位是`m`，则表示以MB为单位，设置`model_gb`为1
        model_gb = 1
    else:
        model_gb = float(model_size[:-1])               # 否则，提取模型大小的数值部分，作为`model_gb`
    if model_gb < 20:                                   # 根据模型大小设置并行计算资源的数量
        n_gpu = 1
    elif model_gb < 50:
        n_gpu = 4
    else:
        n_gpu = 8
    args.parallel = n_gpu > 1                           # 如果并行计算资源的数量大于1，则设置`parallel`为`True`
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)     # 获取环境变量`CUDA_VISIBLE_DEVICES`
    if isinstance(cuda_visible_devices, str):           # 如果环境变量`CUDA_VISIBLE_DEVICES`是字符串，则按逗号分隔为列表
        cuda_visible_devices = cuda_visible_devices.split(",")
    else:
        cuda_visible_devices = list(range(8))           # 否则，设置为0-7的列表
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(      # 设置环境变量`CUDA_VISIBLE_DEVICES`为前`n_gpu`个GPU设备
        [str(dev) for dev in cuda_visible_devices[:n_gpu]]
    )
    print("CUDA_VISIBLE_DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])
    return cuda_visible_devices

import torch
import accelerate


def get_module_by_name_suffix(model, module_name: str):
    """
    在给定模型中，根据模型名称的后缀查找并返回对应的子模块。

    Args:
        model: 给定的模型。
        module_name: 目标子模块名称的后缀。
    """
    for name, module in model.named_modules():
        if name.endswith(module_name):
            return module


def simple_dispatch_model(model, device_map):
    """
    根据提供的设备映射，将模型的各个子模块分配到不同的设备上。

    Args:
        model: 要分配设备的模型。
        device_map: 设备映射，字典形式，键为子模块名称，值为目标设备名称。
    """
    from accelerate.hooks import add_hook_to_module, AlignDevicesHook

    if "" in device_map:                        # 如果设备映射中有空字符串键，则将其作为默认设备
        d = device_map[""]                      # 获取默认设备
        model = model.to(torch.device(d))       # 将模型移动到默认设备
        model.hf_device_map = device_map        # 设置模型的设备映射
        return model

    tied_params = accelerate.utils.modeling.find_tied_parameters(model)     # 查找模型中的绑定参数，并存储在`tied_params`中，以便后续重新绑定，例如共享权重的层
    if set(device_map.values()) == {"cpu"} or set(device_map.values()) == { # 如果设备映射的值全为`cpu`或`disk`，则将主设备设置为`cpu`
        "cpu",
        "disk",
    }:
        main_device = "cpu"
    else:                                                                   # 否则，将主设备设置为设备映射中的第一个非`cpu`和`disk`设备
        main_device = [d for d in device_map.values() if d not in ["cpu", "disk"]][0]

    cpu_offload_group = [(n, d) for n, d in device_map.items() if d == "cpu"]   # 构建包含需要卸载到CPU的子模块名称和设备的列表
    prev_hook = None                                                        # 用于在添加多个钩子时，保持前后钩子的链接关系
    for idx, (n, d) in enumerate(cpu_offload_group):                        # 遍历需要卸载到CPU的子模块
        m = get_module_by_name_suffix(model, n)                             # 根据子模块名称`n`获取对应的子模块
        _, prev_hook = accelerate.cpu_offload_with_hook(                    # 将子模块`m`卸载到CPU，同时关联前一个钩子`prev_hook`，并更新`prev_hook`为当前钩子
            m, execution_device=main_device, prev_module_hook=prev_hook
        )
    # set first cpu offload module's prev_module_hook to the last cpu offload module's hook
    if len(cpu_offload_group) > 1:
        get_module_by_name_suffix(
            model, cpu_offload_group[0][0]                                  # 获取第一个需要卸载到CPU的子模块
        )._hf_hook.prev_module_hook = prev_hook                             # 将其前一个钩子设置为最后一个卸载到CPU的子模块的钩子，实现钩子的链式连接

    for n, d in device_map.items():
        m = get_module_by_name_suffix(model, n)                             # 获取子模块的模块实例
        if d != "cpu":
            d = torch.device(d)                                             # 将设备名称转换为`torch.device`对象
            hook = AlignDevicesHook(d, io_same_device=True, place_submodules=True)  # 创建设备对齐钩子，确保输入输出在指定设备上进行，同时将子模块放置在指定设备上
            add_hook_to_module(m, hook)                                     # 将钩子添加到子模块上，实现设备对齐
    accelerate.utils.modeling.retie_parameters(model, tied_params)          # 重新绑定之前查找的绑在一起的参数，确保参数共享关系在设备调度后仍然保持一致
    model.hf_device_map = device_map                                        # 记录模型的设备映射信息

    return model

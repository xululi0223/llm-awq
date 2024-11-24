def get_op_by_name(module, op_name):
    """
    根据给定的操作名称，在指定的模块中查找并返回对应的子模块。

    Args:
        module: 指定搜索的模块。
        op_name: 操作名称。
    """
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")


def set_op_by_name(layer, name, new_module):
    """
    根据给定的名称路径，在指定的模块中设置或替换对应的子模块。

    Args:
        layer: 指定的模块。
        name: 要设置的操作名称路径，支持嵌套路径。
        new_module: 要设置的新模块。
    """
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):            # 逐层查找并访问子模块
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def get_op_name(module, op):
    """
    根据给定的子模块，在指定的模块中查找并返回对应的操作名称。

    Args:
        module: 指定搜索的模块。
        op: 要查找的子模块实例。
    """
    # get the name of the op relative to the module
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError(f"Cannot find op {op} in module {module}")


def append_str_prefix(x, prefix):
    """
    为输入对象中的所有字符串元素添加前缀，支持嵌套的元组和列表结构。

    Args:
        x: 输入对象，可以是字符串、元组、列表或其他类型。
        prefix: 要添加的前缀字符串。
    """
    if isinstance(x, str):
        return prefix + x
    elif isinstance(x, tuple):
        return tuple([append_str_prefix(y, prefix) for y in x])
    elif isinstance(x, list):
        return [append_str_prefix(y, prefix) for y in x]
    else:
        return x

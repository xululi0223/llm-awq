import math
import torch
import torch.nn as nn
import awq_inference_engine  # with CUDA kernels


def make_divisible(c, divisor):
    """
    确保给定的数能被指定的除数整除。

    Args:
        c: 给定的数。
        divisor: 除数。
    """
    return (c + divisor - 1) // divisor                     # 确保当`c`不能被`divisor`整除时，结果向上取整


def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    """
    根据输入特征数、分组大小和打包数计算出适合的宽度。

    Args:
        in_features: 输入特征数。
        group_size: 分组大小，默认为128。
        pack_num: 打包数，默认为8。
    """
    # 根据`group_size`确定`size_multiplier`
    if group_size >= 128:
        size_multiplier = 1
    elif group_size == 64:
        size_multiplier = 2
    elif group_size == 32:
        size_multiplier = 4
    else:
        raise NotImplementedError

    base_width = make_divisible(in_features // group_size, pack_num)            # 确保每组特征数能被`pack_num`整除
    base_width = make_divisible(base_width, size_multiplier) * size_multiplier  # 确保`base_width`能被`size_multiplier`整除，乘以`size_multiplier`以获得最终的宽度
    return base_width


def pack_intweight(unpacked_qweight, interleave, kstride):
    """
    将未打包的量化权重进行打包处理，以优化内存布局和提高计算效率。
    主要通过重新排列和位操作将量化权重压缩成更紧凑的格式。

    Args:
        unpacked_qweight: 未打包的量化权重张量，形状为[N, K]。
        interleave: 交错数量，用于后续的排列组合。
        kstride: 步幅，用于后续的重排列。
    """
    # unpacked_qweight: [N, K]
    N = unpacked_qweight.shape[0]                       # 获取输出特征数
    K = unpacked_qweight.shape[1]                       # 获取输入特征数

    Packed_Kernel = unpacked_qweight.cpu().numpy().reshape(N, K // 32, 32)      # 重塑`unpacked_qweight`为形状为[N, K // 32, 32]的张量，即每32个输入特征为一组
    # np.arange(32).reshape(4, 4, 2).transpose(1, 0, 2) => [0, 1, 8, 9, 16, 17, 24, 25, ...]
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 3, 2, 4)     # 重塑`Packed_Kernel`为形状为[N, K // 32, 4, 4, 2]的张量，并进行转置操作
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 32)                       # 重塑`Packed_Kernel`回形状为[N, K // 32, 32]的张量

    # reorder each 8 weights for fast dequantization
    # [0, 1, 2, 3, 4, 5, 6, 7] => [0, 2, 4, 6, 1, 3, 5, 7]
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 8)                     # 重塑`Packed_Kernel`为形状为[N, K // 32, 4, 8]的张量，即每8个权重为一组
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 2, 4, 3)     # 重塑`Packed_Kernel`为形状为[N, K // 32, 4, 4, 2]的张量，并进行转置操作，变为[N, K // 32, 4, 2, 4]
    Packed_Kernel = Packed_Kernel.reshape(N, K)                                 # 重塑`Packed_Kernel`回形状为[N, K]的张量，完成权重的重新排列以加快反量化过程

    # interleaving every four rows
    Packed_Kernel = Packed_Kernel.reshape(                                      # 将`Packed_Kernel`重塑为形状为[N // interleave, interleave, K // kstride, kstride]的张量
        N // interleave, interleave, K // kstride, kstride
    )
    # N // 4, K // 64, 4, 64
    Packed_Kernel = Packed_Kernel.transpose(0, 2, 1, 3)                         # 对`Packed_Kernel`进行转置操作，变为[N // interleave, K // kstride, interleave, kstride]
    Packed_Kernel = Packed_Kernel.reshape(                                      # 重塑`Packed_Kernel`为形状为[N // interleave, K // kstride, kstride, interleave]的张量
        N // interleave, K // kstride, kstride, interleave
    )
    # Packing -> (N // 4, K // 64, 64)
    Packed_Kernel = (                                                           # 对最后一个维度的每4个元素进行位操作，将4个元素压缩为一个元素
        Packed_Kernel[..., 0]                                                   # 将第一个元素左移0位
        | (Packed_Kernel[..., 1] << 4)                                          # 将第二个元素左移4位
        | (Packed_Kernel[..., 2] << 8)                                          # 将第三个元素左移8位
        | (Packed_Kernel[..., 3] << 12)                                         # 将第四个元素左移12位
    )                                                                           # 使用按位或操作将这些值合并，生成打包后的数据
    # reshape to (N // 4, K), FP16 format
    Packed_Kernel = Packed_Kernel.reshape(N // interleave, K)                   # 将`Packed_Kernel`重塑为形状为[N // interleave, K]的张量，完成打包操作
    qweight = (                                                                 # 将`Packed_Kernel`转换为`torch.tensor`张量，并转换为`int16`类型
        torch.tensor(Packed_Kernel.astype("int16"))
        .to(unpacked_qweight.device)
        .contiguous()
    )
    return qweight


class ScaledActivation(nn.Module):
    """
    该类用于对激活函数的输出进行缩放处理。
    通过将激活结果除以预定义的缩放因子，实现对激活值的动态调整。
    """
    def __init__(self, module, scales):
        """
        Args:
            module: 激活函数模块。
            scales: 缩放因子。
        """
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)         # 将传入的缩放因子转换为可训练的参数

    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)        # 将激活函数的输出除以缩放因子，实现对激活值的动态调整


class WQLinear(nn.Module):
    """
    该类用于实现4位量化的线性（全连接）层。
    该类通过量化权重和缩放因子来减少模型的内存占用，同时保持性能。
    """
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        """
        Args:
            w_bit: 权重量化的位数，仅支持4位。
            group_size: 分组大小，用于计算适应的宽度。
            in_features: 输入特征数。
            out_features: 输出特征数。
            bias: 是否包含偏置。
            dev: 设备。
        """
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features       # 如果分组大小为-1，则设置为输入特征数
        self.split_k_iters = 8                                                  # 拆分`k`的迭代次数
        self.interleave = 4                                                     # 交错数量
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0                          # 确保输入特征数能被分组大小整除
        assert out_features % (32 // self.w_bit) == 0                           # 确保输出特征数能被（32除以权重位数）整除
        pack_num = 32 // self.w_bit                                             # 计算每组打包的数量
        int16_pack_num = 16 // self.w_bit                                       # 计算每组打包的数量（int16）

        assert out_features % (self.interleave) == 0                            # 确保输出特征数能被交错数量整除
        self.register_buffer(                                                   # 注册缓冲区，用于存储量化后的权重
            "qweight",
            torch.zeros(
                (
                    out_features // self.interleave,
                    in_features // int16_pack_num * self.interleave,
                ),
                dtype=torch.int16,
                device=dev,
            ),
        )
        self.register_buffer(   
            "scales",
            torch.zeros(                                                        # 注册缓冲区，用于存储缩放因子
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )
        self.register_buffer(                                                   # 注册缓冲区，用于存储缩放后的零值
            "scaled_zeros",
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )

        if bias:
            self.register_buffer(                                               # 注册缓冲区，用于存储偏置
                "bias", torch.zeros((out_features), dtype=torch.float16, device=dev)
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None
    ):
        """
        从现有的线性层创建一个4位量化的线性层实例。

        Args:
            linear: 原始的线性层。
            w_bit: 权重量化的位数，仅支持4位。
            group_size: 分组大小，用于计算适应的宽度。
            init_only: 是否仅初始化。
            scales: 缩放因子。
            zeros: 零值。
        """
        # 使用传入的参数创建一个新的4位量化线性层实例
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        if init_only:  # just prepare for loading sd        # 如果仅初始化，则直接返回，适用于仅准备加载状态字典时
            return awq_linear

        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None     # 确保缩放因子和零值不为空
        scale_zeros = zeros * scales                        # 计算缩放后的零值

        pack_num = 32 // awq_linear.w_bit                   # 计算每组打包的数量
        qscales = torch.zeros(                              # 初始化缩放因子
            (
                scales.shape[0],
                calculate_zeros_width(linear.in_features, group_size) * pack_num,
            ),
            dtype=torch.float16,
            device=scales.device,
        )
        qscales[:, : scales.shape[1]] = scales              # 将传入的缩放因子赋值到qscales的前`scales.shape[1]`列
        # awq_linear.scales = scales.clone().half()
        awq_linear.scales = qscales.transpose(1, 0).contiguous()        # 将缩放因子转置后赋值给`awq_linear.scales`
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()    # 如果有偏置，则将偏置拷贝并转换为`half`类型

        intweight = []                                      # 初始化空列表，用于存储量化后的权重
        for idx in range(awq_linear.in_features):           # 遍历输入特征数
            intweight.append(
                torch.round(
                    (linear.weight.data[:, idx] + scale_zeros[:, idx // group_size])
                    / qscales[:, idx // group_size]         # 计算量化权重，并进行四舍五入转换为整数，存储到`intweight`列表中
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)             # 将`intweight`列表中的元素拼接为张量
        # intweight = intweight.t().contiguous()
        intweight = intweight.to(dtype=torch.int32)         # 将`intweight`转换为`int32`类型
        awq_linear.qweight = pack_intweight(                # 对量化后的权重进行打包处理，并赋值给`awq_linear.qweight`
            intweight.contiguous(), interleave=4, kstride=64
        )

        zeros = zeros.to(dtype=torch.int32)                 # 将零值转换为`int32`类型
        scaled_zeros = torch.zeros_like(qscales)            # 初始化缩放后的零值
        # scaled_zeros[:, :scales.shape[1]] = -(qscales[:, :scales.shape[1]] * (zeros.to(torch.float32) - 8.0)).to(torch.float16)
        scaled_zeros[:, : scales.shape[1]] = -(
            qscales[:, : scales.shape[1]] * (zeros.to(torch.float32))   # 计算缩放后的零值，并赋值给`scaled_zeros`的前`scales.shape[1]`列
        ).to(torch.float16)
        awq_linear.scaled_zeros = scaled_zeros.transpose(1, 0).contiguous()     # 转置`scaled_zeros`后赋值给`awq_linear.scaled_zeros`

        return awq_linear

    @torch.no_grad()
    def forward(self, x):
        # out_shape = x.shape[:-1] + (self.out_features,)
        # inputs = x.reshape(-1, x.shape[-1])
        inputs = x
        if inputs.numel() / inputs.shape[-1] < 8:                       # 计算输入张量的总元素数与最后一个维度的壁纸，如果小于8，则调用`gemv_forward_cuda_new`函数，适用于小规模输入
            out = awq_inference_engine.gemv_forward_cuda_new(
                inputs,
                self.qweight,
                self.scales,
                self.scaled_zeros,
                inputs.numel() // inputs.shape[-1],
                self.out_features,
                self.in_features,
                self.group_size,
            )
        else:
            out = awq_inference_engine.gemm_forward_cuda_new(           # 否则，调用`gemm_forward_cuda_new`函数，适用于大规模输入
                inputs, self.qweight, self.scales, self.scaled_zeros
            )  # - 8.0 * self.scales)
        out = out + self.bias if self.bias is not None else out         # 如果有偏置，则将偏置加到输出上
        # print(out)
        # assert 0
        return out

    def extra_repr(self) -> str:
        """
        提供模块的额外字符串表示。
        """
        return (
            "in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.w_bit,
                self.group_size,
            )
        )

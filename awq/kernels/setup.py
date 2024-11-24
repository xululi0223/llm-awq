# find_packages: 自动发现Python包中的所有子包，无需手动列出
# setup: 用于配置包的源数据和构建选项
from setuptools import find_packages, setup
# BuildExtension: 用于编译扩展，用于处理C++和CUDA代码的编译
# CUDAExtension: 专门用于构建包含CUDA代码的扩展模块
# CppExtension: 用于构建仅包含C++代码的扩展模块
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


# 配置编译参数
# 定义C++和NVCC编译器的额外编译参数，以优化编译过程并启用特定的功能
extra_compile_args = {
    "cxx": ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"],      # C++编译器参数
    "nvcc": [                                                                       # NVCC编译器参数
        "-O3",
        "-std=c++17",
        "-DENABLE_BF16",  # TODO
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--threads=8",
    ],
}

# 配置包的构建和安装
setup(
    name="awq_inference_engine",                            # 指定包名
    packages=find_packages(),                               # 自动发现Python包中的所有子包
    ext_modules=[                                           # 定义扩展模块
        CUDAExtension(
            name="awq_inference_engine",
            sources=[
                "csrc/pybind.cpp",                          # 使用PyBind11连接Python和C++代码
                "csrc/quantization/gemm_cuda_gen.cu",
                "csrc/quantization/gemv_cuda.cu",
                "csrc/quantization_new/gemv/gemv_cuda.cu",
                "csrc/quantization_new/gemm/gemm_cuda.cu",
                "csrc/layernorm/layernorm.cu",
                "csrc/position_embedding/pos_encoding_kernels.cu",
                "csrc/attention/ft_attention.cpp",
                "csrc/attention/decoder_masked_multihead_attention.cu",
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},                 # 构建命令
    install_requires=["torch"],                             # 依赖的Python包
)

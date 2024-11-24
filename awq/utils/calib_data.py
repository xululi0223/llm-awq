import torch
from datasets import load_dataset


def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=512):
    """
    从数据集中提取用于校准的数据块。
    Args:
        data: 数据集的名称，目前只支持 "pileval"
        tokenizer: 分词器，用于将文本转换为编码。
        n_samples: 需要处理的样本数量，默认为512。
        block_size: 每个块的大小，默认为512。
    Returns:
        一个列表，包含多个张量，每个张量代表一个数据块。
    """
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    else:
        raise NotImplementedError
    dataset = dataset.shuffle(seed=42)                      # 打乱数据集
    samples = []                                            # 用于存储处理后的样本
    n_run = 0                                               # 记录已处理的样本数量
    for data in dataset:                                    # 遍历数据集中的每一条数据
        line = data["text"]                                 # 提取`text`字段
        line = line.strip()                                 # 去除首尾空格
        line_encoded = tokenizer.encode(line)               # 使用分词器将文本编码成数值表示
        if len(line_encoded) > 512:                         # 如果编码后的长度大于512，则跳过该样本
            continue
        sample = torch.tensor([line_encoded])               # 将编码后的文本转换为张量
        if sample.numel() == 0:                             # 如果张量为空，则跳过该样本
            continue
        samples.append(sample)                              # 将处理后的样本添加到列表中
        n_run += 1                                          # 更新已处理的样本数量
        if n_run == n_samples:                              # 如果已处理的样本数量达到指定数量，则停止处理
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)                 # 将所有样本在维度1上拼接，形成一个大张量
    n_split = cat_samples.shape[1] // block_size            # 计算可以分割成多少个块
    print(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)       # 将大张量按`block_size`分割成多个块
    ]

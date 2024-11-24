import transformers
import torch
from lm_eval.base import BaseLM
import fnmatch


class LMEvalAdaptor(BaseLM):
    """
    继承自`BaseLM`类，用于适配不同的语言模型以符合`lm_eval`框架的要求。
    """
    def __init__(self, model_name, model, tokenizer, batch_size=1, max_length=-1):
        """
        Args:
            model_name: 模型的名称，用于识别和特定处理。
            model: 预训练的语言模型实例。
            tokenizer: 分词器，用于文本与模型输入的转换。
            batch_size: 模型的批处理大小，默认为1。
            max_length: 输入的最大长度，默认为-1，表示自动确定。
        """
        super().__init__()

        assert isinstance(batch_size, int)              # 确保`batch_size`是整数

        self.model_name = model_name
        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer

        # assert isinstance(self.tokenizer, (
        #     transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast,
        #     transformers.T5Tokenizer, transformers.T5TokenizerFast,
        # )), "this tokenizer has not been checked for compatibility yet!"

        self.vocab_size = self.tokenizer.vocab_size     # 记录分词器的词汇表大小

        self._batch_size = batch_size

        self._max_length = max_length

    @property
    def eot_token_id(self):
        """
        返回分词器的特殊文本结束标记`eos_token_id`。
        """
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        """
        确定输入的最大长度。
        """
        if self._max_length != -1:                          # 如果指定了最大长度，则直接返回
            return self._max_length
        if hasattr(self.model.config, "n_ctx"):             # 如果模型配置中有`n_ctx`属性，则返回
            return self.model.config.n_ctx
        elif hasattr(self.model.config, "max_position_embeddings"):     # 如果模型配置中有`max_position_embeddings`属性，则返回
            return self.model.config.max_position_embeddings
        elif hasattr(self.model.config, "n_positions"):     # 如果模型配置中有`n_positions`属性，则返回
            return self.model.config.n_positions
        elif "bloom" in self.model_name:                    # 如果是`bloom`模型，则返回2048
            return 2048
        elif "llama" in self.model_name:                    # 如果是`llama`模型，则返回2048
            return 2048  # TODO: did not check this
        elif "mpt" in self.model_name:                      # 如果是`mpt`模型，则返回2048
            return 2048
        elif "falcon" in self.model_name:                   # 如果是`falcon`模型，则返回2048
            return 2048
        else:
            print(self.model.config)
            raise NotImplementedError

    @property
    def max_gen_toks(self):
        """
        返回生成的最大标记数。
        """
        return 256

    @property
    def batch_size(self):
        """
        返回当前设置的批处理大小。
        """
        return self._batch_size

    @property
    def device(self):
        """
        返回模型运行的设备。
        """
        return "cuda"

    def tok_encode(self, string: str):
        """
        将输入的字符串编码为标记ID列表，不添加特殊标记。
        """
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        """
        将标记ID列表解码回字符串。
        """
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            if isinstance(
                self.model,
                transformers.models.t5.modeling_t5.T5ForConditionalGeneration,          # 如果是`T5ForConditionalGeneration`模型
            ):
                dec_inps = torch.cat(                                                   # 填充decoder_start_token_id，大小为[batch, 1]，与输入拼接
                    [
                        torch.tensor(
                            self.model.generation_config.decoder_start_token_id,
                        )
                        .tile(len(inps), 1)
                        .to(inps),
                        inps,
                    ],
                    dim=1,
                )

                kwargs = {
                    "decoder_input_ids": dec_inps,
                }
            else:
                kwargs = {}
            out = self.model(inps, **kwargs)[0]                                         # 调用模型，传入`inps`和`kwargs`，获取输出的logits
            if (
                "opt" in self.model_name                                                # 如果模型名称中包含`opt`
            ):  # there are a few extra tokens in opt, which we should omit
                return out[:, :, :50257]                                                # 返回`out`的前50257个标记，以去除多余的token
            else:
                return out  # [:, :, :self.tokenizer.vocab_size]

    def _model_generate(self, context, max_length, eos_token_id):
        """
        使用模型的`generate`方法生成文本。

        Args:
            context: 上下文输入张量。
            max_length: 生成的最大长度。
            eos_token_id: 结束标记ID。
        """
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False  # 不进行采样，即进行贪婪生成
        )

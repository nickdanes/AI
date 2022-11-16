from torch import nn
from transformers import BertForSequenceClassification, BertConfig


class BertModel(nn.Module):
    def __init__(self, config):
        super(BertModel, self).__init__()
        # 加载预训练模型
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
        self.device = config.device
        for param in self.bert.parameters():
            # 参数需要梯度
            param.requires_grad = True

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.bert(input_ids=batch_seqs,
                                 attention_mask=batch_seq_masks,
                                 token_type_ids=batch_seq_segments,
                                 return_dict=False,
                                 labels=labels)
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


class BertModelTest(nn.Module):
    def __init__(self, config, model_path):
        super(BertModelTest, self).__init__()
        bert_config = BertConfig.from_pretrained(model_path)
        self.bert = BertForSequenceClassification.from_pretrained(bert_config)
        self.device = config.device

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.bert(input_ids=batch_seqs,
                                 attention_mask=batch_seq_masks,
                                 token_type_ids=batch_seq_segments,
                                 labels=labels)
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities

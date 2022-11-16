import pandas as pd
import torch
from torch.utils.data import Dataset

from args import Config


class DataProcessor(Dataset):
    def __init__(self, bert_tokenizer, path):
        config = Config()
        self.bert_tokenizer = bert_tokenizer
        self.max_seq_len = config.max_seq_len
        self.path = path
        self.seqs, self.seq_masks, self.seq_segments, self.labels = self.load_data(path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.seq_masks[idx], self.seq_segments[idx], self.labels[idx]

    def load_data(self, path):
        df = pd.read_csv(path)
        sentences1 = df['sentence1'].values
        sentences2 = df['sentence2'].values
        labels = df['label'].values
        # 将文字tokenizer
        tokens1 = list(map(self.bert_tokenizer.tokenize, sentences1))
        tokens2 = list(map(self.bert_tokenizer.tokenize, sentences2))
        result = list(map(self.deal_data, tokens1, tokens2))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return torch.Tensor(seqs).type(torch.long), torch.Tensor(seq_masks).type(torch.long), torch.Tensor(
            seq_segments).type(torch.long), torch.Tensor(labels).type(torch.long)

    def deal_data(self, tokens1, tokens2):
        # 长截断
        if len(tokens1) > ((self.max_seq_len - 3) // 2):
            tokens1 = tokens1[0:(self.max_seq_len - 3) // 2]
        if len(tokens2) > ((self.max_seq_len - 3) // 2):
            tokens2 = tokens2[0:(self.max_seq_len - 3) // 2]
        # 首尾中加特殊标记
        seq = ['[CLS]'] + tokens1 + ['[SEP]'] + tokens2 + ['[SEP]']
        # 分别标记AB
        seq_segment = [0] * (len(tokens1) + 2) + [1] * (len(tokens2) + 1)
        # ID化
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (self.max_seq_len - len(seq))
        # 创建seq_mask
        seq_mask = [1] * len(seq) + padding
        # 创建seq_segment
        seq_segment = seq_segment + padding
        # 对seq拼接填充序列
        seq += padding
        return seq, seq_mask, seq_segment

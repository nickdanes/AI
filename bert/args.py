import torch
from transformers import BertTokenizer


class Config(object):

    def __init__(self):
        self.model_name = "Bert"
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
        self.batch_size = 64
        self.max_seq_len = 100
        self.lr = 2e-05
        self.epochs = 3
        self.max_grad_norm = 10.0
        # LCQMC数据集
        self.lcqmc_train_path = "./dataset/lcqmc/LCQMC_train.csv"
        self.lcqmc_dev_path = "./dataset/lcqmc/LCQMC_dev.csv"
        self.lcqmc_test_path = "./dataset/lcqmc/LCQMC_test.csv"
        self.lcqmc_model_path = "./model/lcqmcModel/Model"

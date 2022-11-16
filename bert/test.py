import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from args import Config
from data import DataProcessor
from model import BertModelTest


def main():
    config = Config()
    print("\n", "-" * 20, ">加载数据<", "-" * 20)
    bert_tokenizer = BertTokenizer.from_pretrained('dataset/lcqmc/vocab.txt', do_lower_case=True)
    pretrain = torch.load('model1')
    test_data = DataProcessor(bert_tokenizer, config.lcqmc_test_path)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=config.batch_size)

    model = BertModelTest(config, 'model1')
    model.load_state_dict(pretrain.state_dict())
    model.to(config.device)
    print("\n", "-" * 20, ">测试数据<", "-" * 20)
    accuracy, auc = test(model, test_loader)
    print("\naccuracy: {:.4f}%, auc: {:.4f}\n".format((accuracy * 100), auc))


def test(model, dataloader):
    # Switch the model to eval mode.
    model.eval()
    device = model.device
    accuracy = 0.0
    all_prob = []
    all_labels = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in dataloader:
            # Move input and output data to the GPU if one is used.
            seqs, masks, segments, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(
                device), batch_labels.to(device)
            _, _, probabilities = model(seqs, masks, segments, labels)
            _, out_classes = probabilities.max(dim=1)
            correct = (out_classes == labels).sum()
            accuracy += correct.item()
            all_prob.extend(probabilities[:, 1].cpu().numpy())
            all_labels.extend(batch_labels)
    accuracy /= (len(dataloader.dataset))
    return accuracy, roc_auc_score(all_labels, all_prob)
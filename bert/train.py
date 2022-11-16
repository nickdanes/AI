import os
import torch
from model import BertModel
from args import Config
from data import DataProcessor

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def main(train_dir, val_dir, save_dir):
    # 模型保存路径
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    config = Config()

    # 加载数据
    print("\n", "-" * 20, ">加载数据<", "-" * 20)
    train_data = DataProcessor(config.bert_tokenizer, train_dir)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=config.batch_size)

    val_data = DataProcessor(config.bert_tokenizer, val_dir)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=config.batch_size)

    model = BertModel(config)
    model.to(config.device)

    # 优化参数
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.85, patience=0)

    # 开始训练
    print("\n", "-" * 20, ">开始训练<", "-" * 20)
    for epoch in range(0, config.epochs):
        epoch_loss, epoch_accuracy = train(model, train_loader, optimizer, config.max_grad_norm)
        print("* Train epoch {}:loss = {:.4f}, accuracy: {:.4f}%".format(epoch + 1, epoch_loss, (epoch_accuracy * 100)))

        epoch_loss, epoch_accuracy, epoch_auc = validate(model, val_loader)
        print("* Valid epoch {}:loss: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}".format(epoch+1, epoch_loss,
                                                                                     (epoch_accuracy * 100), epoch_auc))

        scheduler.step(epoch_accuracy)
    torch.save(model, 'model1')


def train(model, dataloader, optimizer, max_gradient_norm):
    # Switch the model to train mode.
    model.train()
    device = model.device
    running_loss = 0.0
    correct_preds = 0
    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in enumerate(tqdm_batch_iterator):
        # Move input and output data to the GPU if it is used.
        seqs, masks, segments, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(
            device), batch_labels.to(device)
        optimizer.zero_grad()
        loss, logits, probabilities = model(seqs, masks, segments, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        running_loss += loss.item()

        _, out_classes = probabilities.max(dim=1)
        correct = (out_classes == labels).sum()
        correct_preds += correct.item()
        description = "batch loss: {:.4f}".format(running_loss / (batch_index + 1))
        tqdm_batch_iterator.set_description(description)
    loss = running_loss / len(dataloader)
    accuracy = correct_preds / len(dataloader.dataset)
    return loss, accuracy


def validate(model, dataloader):
    # Switch to evaluate mode.
    model.eval()
    device = model.device
    running_loss = 0.0
    running_accuracy = 0.0
    all_prob = []
    all_labels = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in dataloader:
            # Move input and output data to the GPU if one is used.
            seqs = batch_seqs.to(device)
            masks = batch_seq_masks.to(device)
            segments = batch_seq_segments.to(device)
            labels = batch_labels.to(device)
            loss, logits, probabilities = model(seqs, masks, segments, labels)
            running_loss += loss.item()
            _, out_classes = probabilities.max(dim=1)
            correct = (out_classes == labels).sum()
            running_accuracy += correct.item()
            all_prob.extend(probabilities[:, 1].cpu().numpy())
            all_labels.extend(batch_labels)
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))
    return epoch_loss, epoch_accuracy, roc_auc_score(all_labels, all_prob)

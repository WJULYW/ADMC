import torch

# torch.manual_seed(10)
# torch.cuda.manual_seed_all(10)
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import random
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from model.Encoders import QEncoder, AEncoder
from configure.config import config


class pure_model(nn.Module):
    def __init__(self, pretrain_model=config['pretrain_model'], output_class=config['feature_dim']):
        super(pure_model, self).__init__()
        self.QEncoder = QEncoder(isPretrain=False, model=pretrain_model)
        self.AEncoder = AEncoder(isPretrain=False, model=pretrain_model)

        self.mlp = nn.Sequential(
            # nn.Linear(config["seq_feature_dim"] * 2, config["seq_feature_dim"] // 2),
            # nn.ReLU(),
            nn.Linear(config[pretrain_model], output_class))
        self.isTrain = True
        self.criterion = nn.CrossEntropyLoss()

    def predictor_mlp(self, q_feat, label_feat):
        q_feat = q_feat.unsqueeze(1).expand(-1, label_feat.shape[0], -1)
        logits = self.mlp(q_feat)
        return F.sigmoid(logits.squeeze(-1))

    def predictor_mask(self, Q_feat, mask):
        logits = self.mlp(Q_feat)
        return logits * mask

    def forward(self, Q_feat):
        logits = self.mlp(Q_feat)
        return logits

    def interaction(self, q_feat, a_feat, q, a):
        q_feat, a_feat = self.esim_sentence(q_feat, a_feat)

        if not self.isTrain:
            q_feat = q_feat[:, 0, :, :].squeeze(1)
            a_feat = a_feat[0, :, :, :].squeeze(0)
        pooled_q = F.normalize(self.QEncoder.mean_pooling(q_feat, q['attention_mask']), p=2, dim=-1)
        pooled_a = F.normalize(self.AEncoder.mean_pooling(a_feat, a['attention_mask']), p=2,
                               dim=-1)
        if not self.isTrain:
            pooled_q = pooled_q.expand(pooled_a.shape[0], -1)
        return F.normalize(torch.cat([pooled_q, pooled_a], dim=-1), dim=-1)

    def esim_sentence(self, q_feat, a_feat):
        if not self.isTrain:
            q_feat = q_feat.unsqueeze(1).expand(-1, a_feat.shape[0], -1, -1)
            a_feat = a_feat.unsqueeze(0).expand(q_feat.shape[0], -1, -1, -1)
        a_alpha = F.softmax(torch.matmul(a_feat, q_feat.transpose(-2, -1).contiguous()), dim=-1)
        # q_feat_i = torch.matmul(q_alpha, a_feat)
        a_feat_i = torch.matmul(a_alpha, q_feat)
        return q_feat, a_feat_i
        # return F.normalize(torch.cat([q_feat, a_feat_i], dim=-1), dim=-1)


def attention(k, q):
    score = F.softmax(torch.matmul(k, q.t()).squeeze(-1), dim=-2)
    res = torch.matmul(score, q)
    return res


class ATT_model(nn.Module):
    def __init__(self, pretrain_model='bert-base-uncased', output_class=374):
        super(ATT_model, self).__init__()
        self.QEncoder = QEncoder(isPretrain=False, model=pretrain_model)
        self.AEncoder = AEncoder(isPretrain=False, model=pretrain_model)

        self.mlp = nn.Sequential(
            nn.Linear(config[pretrain_model] * 2, output_class))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, q, label_data):
        Q_feat = self.QEncoder(q)
        # A_feat = self.AEncoder(a)
        # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        label_feat = attention(Q_feat, self.AEncoder(label_data))
        logits = self.mlp(F.normalize(torch.cat([Q_feat, label_feat], dim=-1), dim=-1))
        return logits


class naive_match_model(nn.Module):
    def __init__(self, pretrain_model='sentence-transformers/all-MiniLM-L6-v2'):
        super(naive_match_model, self).__init__()
        self.QEncoder = QEncoder(isPretrain=False, model=pretrain_model)
        self.AEncoder = AEncoder(isPretrain=False, model=pretrain_model)

        self.isTrain = True
        self.criterion = nn.CrossEntropyLoss()

    def predictor_cos(self, q_feat, label_feat):
        q_feat = q_feat.unsqueeze(1).expand(-1, label_feat.shape[0], -1)
        label_feat = label_feat.unsqueeze(0).expand(q_feat.shape[0], -1, -1)
        # score = F.kl_div(q_feat.softmax(dim=-1).log(), label_feat.softmax(dim=-1), reduction='none')
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        score = cos(q_feat, label_feat)
        return score

    def predictor_innerproduct(self, q_feat, label_feat):
        q_feat = q_feat.unsqueeze(1).expand(-1, label_feat.shape[0], -1)
        label_feat = label_feat.unsqueeze(0).expand(q_feat.shape[0], -1, -1)
        score = F.sigmoid(torch.sum(torch.mul(q_feat, label_feat), dim=-1))
        return score

    def predictor_mask(self, q_feat, mask, id2sentence_A, tokenizer):
        label_feat = []
        for idx in range(len(mask)):
            mask_feat = [id2sentence_A[n] for n in mask[idx]]
            mask_feat = tokenizer(mask_feat,
                                  add_special_tokens=True,
                                  max_length=config['max_seq_A_len'],
                                  pad_to_max_length=True,
                                  truncation=True,
                                  return_token_type_ids=True,
                                  return_attention_mask=True,
                                  return_tensors='pt')
            for key in mask_feat.keys():
                mask_feat[key] = mask_feat[key].cuda()
            mask_feat = self.AEncoder(mask_feat)

            label_feat.append(mask_feat.unsqueeze(0))
        label_feat = torch.cat(label_feat, dim=0)
        q_feat = q_feat.unsqueeze(1).expand(-1, label_feat.shape[1], -1)
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        score = torch.exp(cos(q_feat, label_feat))
        return score

    def interaction(self, q_feat, a_feat, q, a):
        q_feat, a_feat = self.esim_sentence(q_feat, a_feat)
        if not self.isTrain:
            q_feat = q_feat[:, 0, :, :].squeeze(1)
            a_feat = a_feat[0, :, :, :].squeeze(0)
        pooled_q = F.normalize(self.QEncoder.mean_pooling(q_feat, q['attention_mask']), p=2,
                               dim=-1)
        pooled_a = F.normalize(self.AEncoder.mean_pooling(a_feat, a['attention_mask']), p=2,
                               dim=-1)
        return pooled_q, pooled_a

    def esim_sentence(self, q_feat, a_feat):
        if not self.isTrain:
            q_feat = q_feat.unsqueeze(1).expand(-1, a_feat.shape[0], -1, -1)
            a_feat = a_feat.unsqueeze(0).expand(q_feat.shape[0], -1, -1, -1)
        q_alpha = F.softmax(torch.matmul(q_feat, a_feat.transpose(-2, -1).contiguous()), dim=-1)
        a_alpha = F.softmax(torch.matmul(a_feat, q_feat.transpose(-2, -1).contiguous()), dim=-1)
        q_feat_i = torch.matmul(q_alpha, a_feat)
        a_feat_i = torch.matmul(a_alpha, q_feat)

        return F.normalize(torch.cat([q_feat, q_feat_i], dim=-1), dim=-1), F.normalize(torch.cat(
            [a_feat, a_feat_i], dim=-1), dim=-1)

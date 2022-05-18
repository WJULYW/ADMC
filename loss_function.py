import numpy as np
import pandas as pd
from torch import optim
import torch
import torch.nn as nn
import random

# torch.manual_seed(1024)
# torch.cuda.manual_seed_all(1024)
# np.random.seed(1024)
from torch.utils import data
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from model.Ensembel import pure_model, naive_match_model
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import glob
from data_prepare import data_for_encoder, data_for_neg, data_augmentation_translation, data_augmentation_Paraphrase
from dataset import pretrainQ_Dataset, A_learn_dataset, match_Dataset, puretrain_dataset, similarity_mask_dataset
from configure.config import config
from tqdm import tqdm


def margine_hinge_loss_cos(anchor, batch, batch_size):
    loss = 0
    cos1 = nn.CosineSimilarity(dim=0, eps=1e-6)
    cos2 = nn.CosineSimilarity(dim=1, eps=1e-6)
    for idx in range(batch_size):
        l = list(range(batch_size))
        del l[idx]
        # num_neg = min(config["negative_label_num"], batch_size - 1)
        num_neg = batch_size - 1
        neg_idx = random.sample(l, num_neg)
        res_pos = F.sigmoid(cos1(anchor[idx], batch[idx])).unsqueeze(0).expand(num_neg, -1)
        res_neg = F.sigmoid(cos2(anchor[idx].unsqueeze(0).expand(num_neg, -1), batch[neg_idx]))
        res = config['margine'] - res_pos + res_neg
        for j in range(num_neg):
            if res[j, 0] < 0:
                res[j, 0] = 0
        if idx == 0:
            loss = torch.sum(res)
        else:
            loss += torch.sum(res)
    return loss



def margine_hinge_loss_inner(anchor, batch, batch_size):
    loss = 0
    for idx in range(batch_size):
        l = list(range(batch_size))
        del l[idx]
        # num_neg = min(config["negative_label_num"], batch_size - 1)
        num_neg = batch_size - 1
        neg_idx = random.sample(l, num_neg)
        res_pos = F.sigmoid(torch.sum(torch.mul(anchor[idx], batch[idx]))).unsqueeze(0).expand(num_neg, -1)
        res_neg = F.sigmoid(torch.sum(torch.mul(anchor[idx].unsqueeze(0).expand(num_neg, -1), batch[neg_idx])))
        res = config['margine'] - res_pos + res_neg
        for j in range(num_neg):
            if res[j, 0] < 0:
                res[j, 0] = 0
        if idx == 0:
            loss = torch.sum(res)
        else:
            loss += torch.sum(res)
    return loss

def margine_hinge_loss_mlp(anchor, batch, batch_size, model):
    loss = 0
    for idx in range(batch_size):
        l = list(range(batch_size))
        del l[idx]
        # num_neg = min(config["negative_label_num"], batch_size - 1)
        num_neg = batch_size - 1
        neg_idx = random.sample(l, num_neg)
        res_pos = F.sigmoid(model(anchor[idx], batch[idx])).unsqueeze(0).expand(num_neg, -1)
        res_neg = F.sigmoid(model(anchor[idx].unsqueeze(0).expand(num_neg, -1), batch[neg_idx]))
        res = config['margine'] - res_pos + res_neg
        for j in range(num_neg):
            if res[j, 0] < 0:
                res[j, 0] = 0
        if idx == 0:
            loss = torch.sum(res)
        else:
            loss += torch.sum(res)
    return loss


def original_bpr_loss_mlp(anchor, batch, batch_size, model):
    loss = 0
    for idx in range(batch_size):
        l = list(range(batch_size))
        del l[idx]
        # num_neg = min(config["negative_label_num"], batch_size - 1)
        num_neg = batch_size - 1
        neg_idx = random.sample(l, num_neg)
        res_pos = model(anchor[idx], batch[idx]).unsqueeze(0).expand(num_neg, -1)
        res_neg = model(anchor[idx].unsqueeze(0).expand(num_neg, -1), batch[neg_idx])
        # res_pos = torch.exp(F.sigmoid(model(anchor[idx], batch[idx])) / config['temperature'])
        # res_neg = torch.exp(F.sigmoid(model(anchor[idx].unsqueeze(0).expand(num_neg, -1), batch[neg_idx])) / config['temperature'])
        # res = torch.log(1 + torch.exp(res_neg - res_pos))
        # res = -1 * torch.log(res_pos / torch.sum(res_neg))
        res = -1 * torch.log(F.sigmoid(res_pos - res_neg))
        if idx == 0:
            loss = torch.sum(res)
        else:
            loss += torch.sum(res)
    return loss

def original_bpr_loss_inner(anchor, batch, batch_size):
    loss = 0
    for idx in range(batch_size):
        l = list(range(batch_size))
        del l[idx]
        # num_neg = min(config["negative_label_num"], batch_size - 1)
        num_neg = batch_size - 1
        neg_idx = random.sample(l, num_neg)
        res_pos = torch.sum(torch.mul(anchor[idx], batch[idx]))
        res_neg = torch.sum(torch.mul(anchor[idx].unsqueeze(0).expand(num_neg, -1), batch[neg_idx]))
        # res_pos = torch.exp(F.sigmoid(model(anchor[idx], batch[idx])) / config['temperature'])
        # res_neg = torch.exp(F.sigmoid(model(anchor[idx].unsqueeze(0).expand(num_neg, -1), batch[neg_idx])) / config['temperature'])
        # res = torch.log(1 + torch.exp(res_neg - res_pos))
        # res = -1 * torch.log(res_pos / torch.sum(res_neg))
        res = -1 * torch.log(F.sigmoid(res_pos - res_neg))
        if idx == 0:
            loss = torch.sum(res)
        else:
            loss += torch.sum(res)
    return loss

def original_bpr_loss_cos(anchor, batch, batch_size):
    loss = 0
    cos1 = nn.CosineSimilarity(dim=0, eps=1e-6)
    cos2 = nn.CosineSimilarity(dim=1, eps=1e-6)
    for idx in range(batch_size):
        l = list(range(batch_size))
        del l[idx]

        num_neg = batch_size - 1
        neg_idx = random.sample(l, num_neg)
        res_pos = cos1(anchor[idx], batch[idx])
        res_neg = cos2(anchor[idx].unsqueeze(0).expand(num_neg, -1), batch[neg_idx])

        res = -1 * torch.log(F.sigmoid(res_pos - res_neg))
        if idx == 0:
            loss = torch.sum(res)
        else:
            loss += torch.sum(res)
    return loss


def bpr_loss_inner(anchor, batch, batch_size):
    loss = 0
    for idx in range(batch_size):
        l = list(range(batch_size))
        del l[idx]
        # num_neg = min(config["negative_label_num"], batch_size - 1)
        num_neg = batch_size - 1
        neg_idx = random.sample(l, num_neg)
        #cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        # a=cos(anchor[idx], batch[idx])
        res_pos = torch.exp(F.sigmoid(torch.sum(torch.mul(anchor[idx], batch[idx]))) / config['temperature'])  # .unsqueeze(0).repeat(num_neg, 1)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        res_neg = torch.exp(
            F.sigmoid(torch.sum(torch.mul(anchor[idx].unsqueeze(0).expand(num_neg, -1), batch[neg_idx]), dim=-1)) / config['temperature'])  # .unsqueeze(1)

        res = -1 * torch.log(res_pos / torch.sum(res_neg))
        if idx == 0:
            loss = res
        else:
            loss = loss + res
    return loss

def bpr_loss_mlp(anchor, batch, batch_size, model):
    loss = 0
    for idx in range(batch_size):
        l = list(range(batch_size))
        del l[idx]
        # num_neg = min(config["negative_label_num"], batch_size - 1)
        num_neg = batch_size - 1
        neg_idx = random.sample(l, num_neg)
        #cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        # a=cos(anchor[idx], batch[idx])
        res_pos = torch.exp((F.sigmoid(model(anchor[idx], batch[idx]).unsqueeze(0).expand(num_neg, -1))) / config['temperature'])  # .unsqueeze(0).repeat(num_neg, 1)

        res_neg = torch.exp(
            F.sigmoid(anchor[idx].unsqueeze(0).expand(num_neg, -1), batch[neg_idx], dim=-1) / config['temperature'])  # .unsqueeze(1)

        res = -1 * torch.log(res_pos / torch.sum(res_neg))
        if idx == 0:
            loss = res
        else:
            loss = loss + res
    return loss




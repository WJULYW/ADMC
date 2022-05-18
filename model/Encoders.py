import torch
# torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from tqdm import tqdm
import random
# random.seed(10)
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from configure.config import config


class QEncoder(nn.Module):
    def __init__(self, isPretrain=True, model=config['pretrain_model']):
        super(QEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(model)
        self.isPretrain = isPretrain

    def mean_pooling(self, output, attention_mask):
        token_embeddings = output  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


    def forward(self, x):
        output = self.encoder(**x)
        #a=x['attention_mask']
        # Perform pooling. In this case, mean pooling
        sentence_embeddings = self.mean_pooling(output[0], x['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


class AEncoder(nn.Module):
    def __init__(self, isPretrain=True, model=config['pretrain_model']):
        super(AEncoder, self).__init__()

        self.encoder = AutoModel.from_pretrained(model)
        self.isPretrain = isPretrain

    def mean_pooling(self, output, attention_mask):
        token_embeddings = output  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


    def forward(self, x):
        output = self.encoder(**x)
        # Perform pooling. In this case, mean pooling
        sentence_embeddings = self.mean_pooling(output[0], x['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings



def NT_Xent_loss(anchor, pos, neg):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    fm = torch.sum(torch.exp(cos(anchor.expand(pos.shape[0]+neg.shape[0], -1), torch.cat([pos,neg], dim=0)) / config['temperature']))
    fz = torch.exp(cos(anchor.expand(pos.shape[0], -1), pos) / config['temperature'])
    l = torch.sum(torch.log(torch.div(fz, fm)))
    return -1*l


def pretrain_QEncoder(epoch, model ,train_iter ,optimizer ,config ,device):
    num, loss_ = 0, 0.0

    for i, batch in tqdm(enumerate(train_iter)):
        optimizer.zero_grad()
        num += 1
        loss = 0
        x, y = batch
        for key in x.keys():
            x[key] = x[key].to(device)
        y = y.numpy()
        y_feat = model(x)
        for idx, j in enumerate(y):
            anchor = y_feat[idx,:].unsqueeze(0)
            id_pos = [True if i==j and y_idx!=idx else False for y_idx,i in enumerate(y)]
            pos = y_feat[id_pos,:]
            id_neg = [False if i == j else True for i in y]
            neg = y_feat[id_neg,:]

            # when pos_num in batch is insufficient, add new sample with feature cut-off
            if sum([1 if i==j else 0 for i in y]) < config['pos_num']+1:
                pos_2 = []
                n = config['pos_num'] - sum([1 if i==j else 0 for i in y]) + 1
                for n_i in range(n):
                    cutoff_pos = random.sample(range(config['seq_feature_dim']) , int(config['feat_cutoff_rate']*config['seq_feature_dim']))
                    cutoff_feat = anchor.clone()
                    for position in cutoff_pos:
                        cutoff_feat[:,position] = 0
                    pos_2.append(cutoff_feat)
                pos = torch.cat([pos]+pos_2, dim=0)
            if idx == 0:
                loss = NT_Xent_loss(anchor, pos, neg)
                loss_ += loss.item()
            else:
                l=NT_Xent_loss(anchor, pos, neg)
                loss_ += l.item()
            #loss_ += loss.item()
        #loss = torch.sum(loss)
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
            optimizer.step()
    print("epoch: " + str(epoch) + " average loss: " + str(loss_ / num))
    return loss_ / num










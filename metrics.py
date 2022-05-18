import torch
import math
import numpy as np

def recall_score():
    pass


def get_precision_k(y_pred, y, topk):
    P = []
    for k in topk:
        v, res = torch.topk(y_pred, k, dim=1)
        res = res.cpu().numpy()
        hit = 0
        for i in range(res.shape[0]):
            hit += sum([1 for idx in res[i] if idx == y[i]])
        P.append(hit / (y.shape[0]))
    return P




def get_ndcg_k(y_pred, y, topk):
    NDCG = []
    for k in topk:
        dcg = 0
        _, res = torch.topk(y_pred, k, dim=1)
        res = res.cpu().numpy()

        ndcg = []
        for i in range(res.shape[0]):
            # temp = get_index(y[i],1)
            # temp=6
            idcg = sum([math.log(2) / math.log(i + 2) for i in range(min(temp, k))])
            # print(temp,k)
            dcg = sum([math.log(2) / math.log(j + 2) for j in range(k) if res[i][j] < temp])
            ndcg.append(dcg / idcg)
        NDCG.append(np.mean(ndcg))
    return NDCG

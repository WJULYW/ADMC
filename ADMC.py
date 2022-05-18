import warnings

warnings.filterwarnings('ignore')
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from torch import optim
import torch
import torch.nn as nn
import random

torch.manual_seed(1024)
torch.cuda.manual_seed_all(1024)
np.random.seed(1024)
from torch.utils import data
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from model.Ensembel import pure_model, naive_match_model
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import glob
from data_prepare import data_for_encoder, data_for_neg, data_augmentation_translation, data_augmentation_Paraphrase,find_tail_class
from dataset import pretrainQ_Dataset, A_learn_dataset, match_Dataset, puretrain_dataset, similarity_mask_dataset
from configure.config import config
from tqdm import tqdm


def bpr_loss(anchor, batch, batch_size, implicit=None, y_id=None):
    loss = 0
    count = batch_size
    for idx in range(batch_size):
        l = list(range(batch_size))
        del l[idx]
        # num_neg = min(config["negative_label_num"], batch_size - 1)
        num_neg = batch_size - 1
        neg_idx = random.sample(l, num_neg)
        if implicit is not None and y_id[idx] in implicit.keys():
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            expand_num = implicit[y_id[idx]] + 1
            count += expand_num - 1
            pos = F.dropout(anchor[idx].unsqueeze(0).expand(expand_num, -1), p=config['cut_rate'])
            res_pos = torch.exp(
                cos(pos, batch[idx].unsqueeze(0).expand(expand_num, -1)) / config['temperature'])
            cos = nn.CosineSimilarity(dim=2, eps=1e-6)
            res_neg = torch.exp(
                cos(pos.unsqueeze(1).expand(-1, num_neg, -1), batch[neg_idx].unsqueeze(0).expand(expand_num, -1, -1)) /
                config[
                    'temperature'])
        else:
            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            res_pos = torch.exp(
                cos(anchor[idx], batch[idx]) / config['temperature'])  # .unsqueeze(0).repeat(num_neg, 1)
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            res_neg = torch.exp(
                cos(anchor[idx].unsqueeze(0).expand(num_neg, -1), batch[neg_idx]) / config[
                    'temperature'])  # .unsqueeze(1)

        res = -1 * torch.sum(torch.log(res_pos / torch.sum(res_neg, dim=-1))) if implicit is not None and y_id[
            idx] in implicit.keys() else -1 * torch.log(res_pos / torch.sum(res_neg))
        # res = -1 * torch.log(res_pos / torch.sum(res_neg))
        if idx == 0:
            loss = res
        else:
            loss = loss + res
    return loss


def bpr_mask_loss(model, q_feat, a_feat, a_id, mask, id2sentence_A, tokenizer, batch_size):
    loss = 0
    for idx in range(batch_size):
        neg = list(set(mask[idx]) - set([a_id[idx]]))
        if len(neg) == len(mask[idx]):
            neg = neg[:-1]
        neg = [id2sentence_A[n] for n in neg]
        neg = tokenizer(neg,
                        add_special_tokens=True,
                        max_length=config['max_seq_A_len'],
                        pad_to_max_length=True,
                        truncation=True,
                        return_token_type_ids=True,
                        return_attention_mask=True,
                        return_tensors='pt')
        for key in neg.keys():
            neg[key] = neg[key].to(device)
        neg_feat = model.AEncoder(neg)

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        # a=cos(anchor[idx], batch[idx])
        res_pos = torch.exp(
            cos(q_feat[idx], a_feat[idx]) / config['temperature'])  # .unsqueeze(0).repeat(num_neg, 1)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        res_neg = torch.exp(
            cos(q_feat[idx].unsqueeze(0).expand(len(mask[idx]) - 1, -1), neg_feat) / config[
                'temperature'])  # .unsqueeze(1)
        # res_neg = model(anchor[idx].unsqueeze(0).repeat(num_neg, 1), batch[neg_idx])
        # res = torch.log(1 + torch.exp(res_neg - res_pos))
        res = -1 * torch.log(res_pos / torch.sum(res_neg))
        if idx == 0:
            loss = res
        else:
            loss = loss + res
    return loss


def match_train(epoch, model, train_iter, imbalance_train,optimizer, device):
    model.train()
    num, loss_ = 0, 0.0
    for i, batch in tqdm(enumerate(train_iter)):
        optimizer.zero_grad()
        num += 1
        q, a, y = batch
        # y = y.toarray()
        # label_feat = model.get_label_feat(label)
        for key in q.keys():
            q[key] = q[key].to(device)
            a[key] = a[key].to(device)
        q_feat = model.QEncoder(q)
        a_feat = model.AEncoder(a)
        # q_feat, a_feat = model(q_feat, a_feat)
        loss = bpr_loss(q_feat, a_feat, q_feat.shape[0], imbalance_train, y)

        loss_ += loss.item()
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
            optimizer.step()
    print("epoch: " + str(epoch) + " average matching loss: " + str(loss_ / num))
    return loss_ / num


def match_test(model, test_iter, label_feat, device, epoch):
    model.eval()
    label = []
    pred = []
    ids = []

    # for label in label_data.keys():
    #   label_data[label] = label_data[label].to(device)
    for i, batch in tqdm(enumerate(test_iter)):
        q, a_id = batch
        # y = y.long().to(device)
        # label_feat = model.get_label_feat(label)
        for key in q.keys():
            q[key] = q[key].to(device)
            # a[key] = a[key].to(device)
        q_feat = model.QEncoder(q)
        res = model.predictor_cos(q_feat, label_feat)

        _, id = torch.topk(res, config['topk'], dim=1)
        id = id.cpu().numpy().tolist()
        ids += id

        # for item in id:
        # ids += [id2sentence[j] for j in item]

        pred += torch.argmax(res, dim=1).cpu().numpy().tolist()
        label += a_id.cpu().numpy().tolist()
        # torch.sum(pred).cpu().numpy() / torch.sum(y).cpu().numpy())
    if epoch >= 6:
        print('Precision@10: ', np.mean([1 if label[i] in ids[i] else 0 for i in range(len(label))]))

    print('Accuracy: ', accuracy_score(label, pred) * 100, ', F1 score: ',
          f1_score(label, pred, average='macro') * 100)
    return ids, accuracy_score(label, pred) * 100


def similar_train(epoch, model, train_iter, mask_train, id2sentence_A, tokenizer, optimizer, device):
    num, loss_ = 0, 0.0
    model.train()
    for i, batch in tqdm(enumerate(train_iter)):
        optimizer.zero_grad()
        num += 1
        q, a, mask_id, a_id = batch
        mask_id = mask_id.numpy().tolist()
        a_id = a_id.numpy().tolist()

        for key in q.keys():
            q[key] = q[key].to(device)
            a[key] = a[key].to(device)
        q_feat = model.QEncoder(q)
        a_feat = model.AEncoder(a)
        # q_feat, a_feat = model(q_feat, a_feat)
        mask_batch = [mask_train[idx] for idx in mask_id]
        loss = bpr_mask_loss(model, q_feat, a_feat, a_id, mask_batch, id2sentence_A, tokenizer,
                             q_feat.shape[0])
        loss_ += loss.item()
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
            optimizer.step()
    print("epoch: " + str(epoch) + " average loss: " + str(loss_ / num))
    return loss_ / num


def similar_test(model, test_iter, mask_test, id2sentence_A, tokenizer, device):
    model.eval()
    pred = []
    label = []

    for i, batch in tqdm(enumerate(test_iter)):
        q, mask_id, a_id = batch
        mask_id = mask_id.numpy().tolist()
        mask_batch = [mask_test[idx] for idx in mask_id]
        a_id = a_id.numpy().tolist()
        for key in q.keys():
            q[key] = q[key].to(device)

        res = model.predictor_mask(model.QEncoder(q), mask_batch, id2sentence_A, tokenizer)
        with torch.no_grad():
            id = torch.argmax(res, dim=1).cpu().numpy().tolist()

            for idx, v in enumerate(id):
                pred.append(mask_batch[idx][v])
            label += a_id
    acc = accuracy_score(label, pred) * 100
    f1 = f1_score(label, pred, average='macro') * 100
    print('Accuracy: ', acc, ', F1 score: ', f1)
    return acc,f1

def case_study(model, test_iter, label_feat, device):
    model.eval()
    score_list = []
    label = []
    for i, batch in tqdm(enumerate(test_iter)):
        q, a_id = batch
        for key in q.keys():
            q[key] = q[key].to(device)
        q_feat = model.QEncoder(q)
        res = model.predictor_cos(q_feat, label_feat)
        score_list += res.cpu().detach().numpy().tolist()
        label += a_id.cpu().numpy().tolist()
    return score_list,label



if __name__ == '__main__':
    data_name = config['dataset']
    print('Data Augmentation Method: 1.None 2.Dropout 3.Feature_cut_off 4.Translation 5.Paraphrases')
    idx_data_aug = int(input())
    aug_method = ['None','Dropout','Feature_cut_off','Translation','Paraphrases']
    imbalance_train = None
    if idx_data_aug in [4, 5]:
        if os.path.exists('data/' + data_name + '/' + aug_method[idx_data_aug - 1] + '_' + str(config['Augmentation_target_number']) +
                          '_train_data.csv'):
            train_data = pd.read_csv('data/' + data_name + '/' + aug_method[idx_data_aug - 1]
                                     + '_' + str(config['Augmentation_target_number']) + '_train_data.csv')
            train_data.columns = [0, 1, 'idQ', 'idA']
            test_data = pd.read_csv('data/' + data_name + '/' + aug_method[idx_data_aug - 1] + '_'
                                    + str(config['Augmentation_target_number']) + '_test_data.csv')
            test_data.columns = [0, 1, 'idQ', 'idA']
            A_dict = eval(str(np.load('data/' + data_name + '/' + aug_method[idx_data_aug - 1] + '_'
                                      + str(config['Augmentation_target_number']) + '_A_dict.npy', allow_pickle=True)))
            trainQ_dict = eval(str(np.load('data/' + data_name + '/' + aug_method[idx_data_aug - 1] +
                                           '_' + str(config['Augmentation_target_number']) + '_trainQ_dict.npy', allow_pickle=True)))
            testQ_dict = eval(str(np.load('data/' + data_name + '/' + aug_method[idx_data_aug - 1] + '_'
                                          + str(config['Augmentation_target_number']) + '_testQ_dict.npy', allow_pickle=True)))
        else:
            maxfile = ''
            max = 0
            for filename in glob.glob('data/'+ data_name +'/*.csv'):
                try:
                    tempfile = filename.split('_')
                    if str(tempfile[0].split('\\')[-1]) == str(aug_method[idx_data_aug - 1]):
                        if max < int(tempfile[-3]) and int(tempfile[-3]) < config['Augmentation_target_number']:
                            max = int(tempfile[-3])
                            maxfile = tempfile[0] + '_' + tempfile[1]
                except:
                    pass
            if maxfile == '' or config['Augmentation_target_number'] < max:
                # train = pd.read_csv('data/' + data_name + '/train.tsv', sep='\t', header=None)
                # test = pd.read_csv('data/' + data_name + '/test.tsv', sep='\t', header=None)
                train = pd.read_csv('data/' + data_name + '/train.csv',header=None)
                test = pd.read_csv('data/' + data_name + '/test.csv',header=None)
            else:
                train_data = pd.read_csv(maxfile + '_train_data.csv')
                train_data.columns = [0, 1, 'idQ', 'idA']
                train = train_data.drop(['idQ', 'idA'], axis=1)
                test_data = pd.read_csv(maxfile + '_test_data.csv')
                test_data.columns = [0, 1, 'idQ', 'idA']
                test = test_data.drop(['idQ', 'idA'], axis=1)
            print("Number of data for current dataset file:",len(train))
            if idx_data_aug == 4:
                train = data_augmentation_translation(train, config['Augmentation_target_number'])
            else:
                train = data_augmentation_Paraphrase(train,config['Augmentation_target_number'])
            print("Number of data after data augmentation:",len(train))
            train_data, trainQ_dict, A_dict = data_for_encoder(train, {})
            test_data, testQ_dict, A_dict = data_for_encoder(test, A_dict, False)
            train_data.to_csv(
                'data/' + data_name + '/' + aug_method[idx_data_aug - 1] + '_' + str(config['Augmentation_target_number']) + '_train_data.csv',
                index=False)
            test_data.to_csv(
                'data/' + data_name + '/' + aug_method[idx_data_aug - 1] + '_' + str(config['Augmentation_target_number']) + '_test_data.csv',
                index=False)
            np.save('data/' + data_name + '/' + aug_method[idx_data_aug - 1] + '_' + str(config['Augmentation_target_number']) + '_A_dict', A_dict)
            np.save('data/' + data_name + '/' + aug_method[idx_data_aug - 1] + '_' + str(config['Augmentation_target_number']) + '_trainQ_dict',
                    trainQ_dict)
            np.save('data/' + data_name + '/' + aug_method[idx_data_aug - 1] + '_' + str(config['Augmentation_target_number']) + '_testQ_dict',
                    testQ_dict)
    else:
        if idx_data_aug != 3 and os.path.exists('data/' + data_name + '/train_data.csv'):
            train_data = pd.read_csv('data/' + data_name + '/train_data.csv')
            train_data.columns = [0, 1, 'idQ', 'idA']
            test_data = pd.read_csv('data/' + data_name + '/test_data.csv')
            test_data.columns = [0, 1, 'idQ', 'idA']
            A_dict = eval(str(np.load('data/' + data_name + '/A_dict.npy',
                                      allow_pickle=True)))
            trainQ_dict = eval(
                str(np.load('data/' + data_name + '/trainQ_dict.npy',
                            allow_pickle=True)))
            testQ_dict = eval(str(np.load('data/' + data_name + '/testQ_dict.npy',
                                          allow_pickle=True)))
        elif idx_data_aug == 3 and os.path.exists('data/' + data_name + '/train_data.csv') and os.path.exists('data/' + data_name + '/imbalance_train.npy'):
            train_data = pd.read_csv('data/' + data_name + '/train_data.csv')
            train_data.columns = [0, 1, 'idQ', 'idA']
            test_data = pd.read_csv('data/' + data_name + '/test_data.csv')
            test_data.columns = [0, 1, 'idQ', 'idA']
            A_dict = eval(str(np.load('data/' + data_name + '/A_dict.npy', allow_pickle=True)))
            imbalance_train = eval(str(np.load('data/' + data_name + '/imbalance_train.npy', allow_pickle=True)))
            trainQ_dict = eval(str(np.load('data/' + data_name + '/trainQ_dict.npy', allow_pickle=True)))
            testQ_dict = eval(str(np.load('data/' + data_name + '/testQ_dict.npy', allow_pickle=True)))
        else:
            # train = pd.read_csv('data/' + data_name + '/train.tsv', sep='\t', header=None)
            # test = pd.read_csv('data/' + data_name + '/test.tsv', sep='\t', header=None)
            train = pd.read_csv('data/' + data_name + '/train.csv',header=None)
            test = pd.read_csv('data/' + data_name + '/test.csv',header=None)
            if idx_data_aug == 3:
                train_data, trainQ_dict, A_dict, imbalance_train = data_for_encoder(train, {}, implicit=True)
                np.save('data/' + data_name + '/imbalance_train', imbalance_train)
            else:
                train_data, trainQ_dict, A_dict = data_for_encoder(train, {})
            test_data, testQ_dict, A_dict = data_for_encoder(test, A_dict, False)
            train_data.to_csv('data/' + data_name + '/train_data.csv', index=False)
            test_data.to_csv('data/' + data_name + '/test_data.csv', index=False)
            np.save('data/' + data_name + '/A_dict', A_dict)
            np.save('data/' + data_name + '/trainQ_dict', trainQ_dict)
            np.save('data/' + data_name + '/testQ_dict', testQ_dict)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config['pretrain_model'])

    id2sentence_A = {v: k for k, v in A_dict.items()}

    # #######################################################
    # case_study_data = test_data.sample(n=20)
    # case_study_data.index = range(0, len(case_study_data))
    # ################################################

    acc1_list = []
    acc2_list = []
    f1_list = []
    for i in range(1):
        best_acc1 = 0
        best_acc2 = 0
        best_f1 = 0

        train_dataset = puretrain_dataset(train_data, tokenizer)
        train_data4mask = pretrainQ_Dataset(train_data, tokenizer)
        test_dataset = pretrainQ_Dataset(test_data, tokenizer)

        # case_study_data = pretrainQ_Dataset(case_study_data,tokenizer)
        # case_study_iter = data.DataLoader(
        # dataset=case_study_data,
        # batch_size=config['test_batch_size'],
        # shuffle=True,
        # num_workers=2)
		# #######################################################

        train_iter = data.DataLoader(
            dataset=train_dataset,
            batch_size=config['recall_train_batch_size'],
            shuffle=True,
            num_workers=2)
        train_iter4mask = data.DataLoader(
            dataset=train_data4mask,
            batch_size=config['recall_train_batch_size'],
            shuffle=False,
            num_workers=2)
        test_iter = data.DataLoader(
            dataset=test_dataset,
            batch_size=config['test_batch_size'],
            shuffle=False,
            num_workers=2)


        model = naive_match_model(pretrain_model= config['pretrain_model'])
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=config["learn_rate"])
        scheduler = StepLR(optimizer, step_size=config["lr_dc_step"], gamma=config["lr_dc"])

        # for test
        label_data = A_learn_dataset(list(A_dict.keys()), tokenizer)
        for label in label_data.keys():
            label_data[label] = label_data[label].to(device)
        print()
        print("Recall train start")
		#######################################################
       	# or_cos,or_idx = case_study(model,case_study_iter,model.AEncoder(label_data),device)
        # cos_list1 = []
        # cos_list2 = []
        # #######################################################
        for epoch in range(1, config['train_epoch'] + 1):
            match_train(epoch, model, train_iter, imbalance_train, optimizer, device)
            scheduler.step(epoch=epoch)
            if epoch % 1 == 0:
                print('Test result in epoch', epoch)
                _,tempacc = match_test(model, test_iter, model.AEncoder(label_data), device, epoch)
                # if tempacc > best_acc1:
                #     cos_list1, _ = case_study(model, case_study_iter, model.AEncoder(label_data), device)
                best_acc1 =  tempacc if tempacc > best_acc1 else best_acc1
        acc1_list.append(best_acc1)
        label_feat = model.AEncoder(label_data)
        print()
        print("Generating candidate label set for classification")
        print("Test Dataset result")
        mask_test,_ = match_test(model, test_iter, label_feat, device, epoch)
        print("Train Dataset result")
        mask_train,_ = match_test(model, train_iter4mask, label_feat, device, epoch)

        # Classification
        classes = [x for x in range(0, len(A_dict))]

        train_dataset = similarity_mask_dataset(train_data, range(len(mask_train)), tokenizer, istrain=True)
        test_dataset = similarity_mask_dataset(test_data, range(len(mask_test)), tokenizer, istrain=False)
        train_iter = data.DataLoader(
            dataset=train_dataset,
            batch_size=config['classification_train_batch_size'],
            shuffle=True,
            num_workers=2)
        test_iter = data.DataLoader(
            dataset=test_dataset,
            batch_size=config['test_batch_size'],
            shuffle=True,
            num_workers=2)
        optimizer = optim.AdamW(model.parameters(), lr=config["s2_learn_rate"])
        scheduler = StepLR(optimizer, step_size=config["s2_lr_dc_step"], gamma=config["s2_lr_dc"])
        print()
        print("Classification train start")
        max = 0
        for filename in glob.glob('data/'+config['dataset']+'/*.pth'):
            # print(filename)
            if max < float(filename.split('\\')[-1].split('_')[1]):
                max = float(filename.split('\\')[-1].split('_')[1])
        max=0
        for epoch in range(1, config['s2_train_epoch']):
            similar_train(epoch, model, train_iter, mask_train, id2sentence_A, tokenizer, optimizer, device)
            scheduler.step(epoch=epoch)
            if epoch % 1 == 0:

                print('Test result in epoch', epoch)
                tempacc,tempf1 = similar_test(model, test_iter, mask_test, id2sentence_A, tokenizer, device)
                # if tempacc > best_acc2:
                #     cos_list2, __ = case_study(model, case_study_iter, model.AEncoder(label_data), device)
                if tempacc > max:
                    max = tempacc
                    torch.save(model.state_dict(), 'data/'+config['dataset']+'/acc_' + str(tempacc) + '_f1_' + str(tempf1) + '.pth')
                best_acc2 =  import warnings

warnings.filterwarnings('ignore')
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from torch import optim
import torch
import torch.nn as nn
import random

torch.manual_seed(1024)
torch.cuda.manual_seed_all(1024)
np.random.seed(1024)
from torch.utils import data
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from model.Ensembel import pure_model, naive_match_model
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import glob
from data_prepare import data_for_encoder, data_for_neg, data_augmentation_translation, data_augmentation_Paraphrase,find_tail_class
from dataset import pretrainQ_Dataset, A_learn_dataset, match_Dataset, puretrain_dataset, similarity_mask_dataset
from configure.config import config
from tqdm import tqdm


def bpr_loss(anchor, batch, batch_size, implicit=None, y_id=None):
    loss = 0
    count = batch_size
    for idx in range(batch_size):
        l = list(range(batch_size))
        del l[idx]
        # num_neg = min(config["negative_label_num"], batch_size - 1)
        num_neg = batch_size - 1
        neg_idx = random.sample(l, num_neg)
        if implicit is not None and y_id[idx] in implicit.keys():
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            expand_num = implicit[y_id[idx]] + 1
            count += expand_num - 1
            pos = F.dropout(anchor[idx].unsqueeze(0).expand(expand_num, -1), p=config['cut_rate'])
            res_pos = torch.exp(
                cos(pos, batch[idx].unsqueeze(0).expand(expand_num, -1)) / config['temperature'])
            cos = nn.CosineSimilarity(dim=2, eps=1e-6)
            res_neg = torch.exp(
                cos(pos.unsqueeze(1).expand(-1, num_neg, -1), batch[neg_idx].unsqueeze(0).expand(expand_num, -1, -1)) /
                config[
                    'temperature'])
        else:
            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            res_pos = torch.exp(
                cos(anchor[idx], batch[idx]) / config['temperature'])  # .unsqueeze(0).repeat(num_neg, 1)
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            res_neg = torch.exp(
                cos(anchor[idx].unsqueeze(0).expand(num_neg, -1), batch[neg_idx]) / config[
                    'temperature'])  # .unsqueeze(1)

        res = -1 * torch.sum(torch.log(res_pos / torch.sum(res_neg, dim=-1))) if implicit is not None and y_id[
            idx] in implicit.keys() else -1 * torch.log(res_pos / torch.sum(res_neg))
        # res = -1 * torch.log(res_pos / torch.sum(res_neg))
        if idx == 0:
            loss = res
        else:
            loss = loss + res
    return loss


def bpr_mask_loss(model, q_feat, a_feat, a_id, mask, id2sentence_A, tokenizer, batch_size):
    loss = 0
    for idx in range(batch_size):
        neg = list(set(mask[idx]) - set([a_id[idx]]))
        if len(neg) == len(mask[idx]):
            neg = neg[:-1]
        neg = [id2sentence_A[n] for n in neg]
        neg = tokenizer(neg,
                        add_special_tokens=True,
                        max_length=config['max_seq_A_len'],
                        pad_to_max_length=True,
                        truncation=True,
                        return_token_type_ids=True,
                        return_attention_mask=True,
                        return_tensors='pt')
        for key in neg.keys():
            neg[key] = neg[key].to(device)
        neg_feat = model.AEncoder(neg)

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        # a=cos(anchor[idx], batch[idx])
        res_pos = torch.exp(
            cos(q_feat[idx], a_feat[idx]) / config['temperature'])  # .unsqueeze(0).repeat(num_neg, 1)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        res_neg = torch.exp(
            cos(q_feat[idx].unsqueeze(0).expand(len(mask[idx]) - 1, -1), neg_feat) / config[
                'temperature'])  # .unsqueeze(1)
        # res_neg = model(anchor[idx].unsqueeze(0).repeat(num_neg, 1), batch[neg_idx])
        # res = torch.log(1 + torch.exp(res_neg - res_pos))
        res = -1 * torch.log(res_pos / torch.sum(res_neg))
        if idx == 0:
            loss = res
        else:
            loss = loss + res
    return loss


def match_train(epoch, model, train_iter, imbalance_train,optimizer, device):
    model.train()
    num, loss_ = 0, 0.0
    for i, batch in tqdm(enumerate(train_iter)):
        optimizer.zero_grad()
        num += 1
        q, a, y = batch
        # y = y.toarray()
        # label_feat = model.get_label_feat(label)
        for key in q.keys():
            q[key] = q[key].to(device)
            a[key] = a[key].to(device)
        q_feat = model.QEncoder(q)
        a_feat = model.AEncoder(a)
        # q_feat, a_feat = model(q_feat, a_feat)
        loss = bpr_loss(q_feat, a_feat, q_feat.shape[0], imbalance_train, y)

        loss_ += loss.item()
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
            optimizer.step()
    print("epoch: " + str(epoch) + " average matching loss: " + str(loss_ / num))
    return loss_ / num


def match_test(model, test_iter, label_feat, device, epoch):
    model.eval()
    label = []
    pred = []
    ids = []

    # for label in label_data.keys():
    #   label_data[label] = label_data[label].to(device)
    for i, batch in tqdm(enumerate(test_iter)):
        q, a_id = batch
        # y = y.long().to(device)
        # label_feat = model.get_label_feat(label)
        for key in q.keys():
            q[key] = q[key].to(device)
            # a[key] = a[key].to(device)
        q_feat = model.QEncoder(q)
        res = model.predictor_cos(q_feat, label_feat)

        _, id = torch.topk(res, config['topk'], dim=1)
        id = id.cpu().numpy().tolist()
        ids += id

        # for item in id:
        # ids += [id2sentence[j] for j in item]

        pred += torch.argmax(res, dim=1).cpu().numpy().tolist()
        label += a_id.cpu().numpy().tolist()
        # torch.sum(pred).cpu().numpy() / torch.sum(y).cpu().numpy())
    if epoch >= 6:
        print('Precision@10: ', np.mean([1 if label[i] in ids[i] else 0 for i in range(len(label))]))

    print('Accuracy: ', accuracy_score(label, pred) * 100, ', F1 score: ',
          f1_score(label, pred, average='macro') * 100)
    return ids, accuracy_score(label, pred) * 100


def similar_train(epoch, model, train_iter, mask_train, id2sentence_A, tokenizer, optimizer, device):
    num, loss_ = 0, 0.0
    model.train()
    for i, batch in tqdm(enumerate(train_iter)):
        optimizer.zero_grad()
        num += 1
        q, a, mask_id, a_id = batch
        mask_id = mask_id.numpy().tolist()
        a_id = a_id.numpy().tolist()

        for key in q.keys():
            q[key] = q[key].to(device)
            a[key] = a[key].to(device)
        q_feat = model.QEncoder(q)
        a_feat = model.AEncoder(a)
        # q_feat, a_feat = model(q_feat, a_feat)
        mask_batch = [mask_train[idx] for idx in mask_id]
        loss = bpr_mask_loss(model, q_feat, a_feat, a_id, mask_batch, id2sentence_A, tokenizer,
                             q_feat.shape[0])
        loss_ += loss.item()
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
            optimizer.step()
    print("epoch: " + str(epoch) + " average loss: " + str(loss_ / num))
    return loss_ / num


def similar_test(model, test_iter, mask_test, id2sentence_A, tokenizer, device):
    model.eval()
    pred = []
    label = []

    for i, batch in tqdm(enumerate(test_iter)):
        q, mask_id, a_id = batch
        mask_id = mask_id.numpy().tolist()
        mask_batch = [mask_test[idx] for idx in mask_id]
        a_id = a_id.numpy().tolist()
        for key in q.keys():
            q[key] = q[key].to(device)

        res = model.predictor_mask(model.QEncoder(q), mask_batch, id2sentence_A, tokenizer)
        with torch.no_grad():
            id = torch.argmax(res, dim=1).cpu().numpy().tolist()

            for idx, v in enumerate(id):
                pred.append(mask_batch[idx][v])
            label += a_id
    acc = accuracy_score(label, pred) * 100
    f1 = f1_score(label, pred, average='macro') * 100
    print('Accuracy: ', acc, ', F1 score: ', f1)
    return acc,f1

def case_study(model, test_iter, label_feat, device):
    model.eval()
    score_list = []
    label = []
    for i, batch in tqdm(enumerate(test_iter)):
        q, a_id = batch
        for key in q.keys():
            q[key] = q[key].to(device)
        q_feat = model.QEncoder(q)
        res = model.predictor_cos(q_feat, label_feat)
        score_list += res.cpu().detach().numpy().tolist()
        label += a_id.cpu().numpy().tolist()
    return score_list,label



if __name__ == '__main__':
    data_name = config['dataset']
    print('Test or Train: 1.Test 2.Train')
    if int(input())==2:
      istest=False
    else:
      istest=True
      print('Please input the name of checkpoint file you want to load:')
      checkpoint=input()
    print('Data Augmentation Method: 1.None 2.Dropout 3.Feature_cut_off 4.Translation 5.Paraphrases')
    idx_data_aug = int(input())
    aug_method = ['None','Dropout','Feature_cut_off','Translation','Paraphrases']
    imbalance_train = None
    print('Data preparing......')
    if idx_data_aug in [4, 5]:
        if os.path.exists('data/' + data_name + '/' + aug_method[idx_data_aug - 1] + '_' + str(config['Augmentation_target_number']) +
                          '_train_data.csv'):
            train_data = pd.read_csv('data/' + data_name + '/' + aug_method[idx_data_aug - 1]
                                     + '_' + str(config['Augmentation_target_number']) + '_train_data.csv')
            train_data.columns = [0, 1, 'idQ', 'idA']
            test_data = pd.read_csv('data/' + data_name + '/' + aug_method[idx_data_aug - 1] + '_'
                                    + str(config['Augmentation_target_number']) + '_test_data.csv')
            test_data.columns = [0, 1, 'idQ', 'idA']
            A_dict = eval(str(np.load('data/' + data_name + '/' + aug_method[idx_data_aug - 1] + '_'
                                      + str(config['Augmentation_target_number']) + '_A_dict.npy', allow_pickle=True)))
            trainQ_dict = eval(str(np.load('data/' + data_name + '/' + aug_method[idx_data_aug - 1] +
                                           '_' + str(config['Augmentation_target_number']) + '_trainQ_dict.npy', allow_pickle=True)))
            testQ_dict = eval(str(np.load('data/' + data_name + '/' + aug_method[idx_data_aug - 1] + '_'
                                          + str(config['Augmentation_target_number']) + '_testQ_dict.npy', allow_pickle=True)))
        else:
            maxfile = ''
            max = 0
            for filename in glob.glob('data/'+ data_name +'/*.csv'):
                try:
                    tempfile = filename.split('_')
                    if str(tempfile[0].split('\\')[-1]) == str(aug_method[idx_data_aug - 1]):
                        if max < int(tempfile[-3]) and int(tempfile[-3]) < config['Augmentation_target_number']:
                            max = int(tempfile[-3])
                            maxfile = tempfile[0] + '_' + tempfile[1]
                except:
                    pass
            if maxfile == '' or config['Augmentation_target_number'] < max:
                # train = pd.read_csv('data/' + data_name + '/train.tsv', sep='\t', header=None)
                # test = pd.read_csv('data/' + data_name + '/test.tsv', sep='\t', header=None)
                train = pd.read_csv('data/' + data_name + '/train.csv',header=None)
                test = pd.read_csv('data/' + data_name + '/test.csv',header=None)
            else:
                train_data = pd.read_csv(maxfile + '_train_data.csv')
                train_data.columns = [0, 1, 'idQ', 'idA']
                train = train_data.drop(['idQ', 'idA'], axis=1)
                test_data = pd.read_csv(maxfile + '_test_data.csv')
                test_data.columns = [0, 1, 'idQ', 'idA']
                test = test_data.drop(['idQ', 'idA'], axis=1)
            print("Number of data for current dataset file:",len(train))
            if idx_data_aug == 4:
                train = data_augmentation_translation(train, config['Augmentation_target_number'])
            else:
                train = data_augmentation_Paraphrase(train,config['Augmentation_target_number'])
            print("Number of data after data augmentation:",len(train))
            train_data, trainQ_dict, A_dict = data_for_encoder(train, {})
            test_data, testQ_dict, A_dict = data_for_encoder(test, A_dict, False)
            train_data.to_csv(
                'data/' + data_name + '/' + aug_method[idx_data_aug - 1] + '_' + str(config['Augmentation_target_number']) + '_train_data.csv',
                index=False)
            test_data.to_csv(
                'data/' + data_name + '/' + aug_method[idx_data_aug - 1] + '_' + str(config['Augmentation_target_number']) + '_test_data.csv',
                index=False)
            np.save('data/' + data_name + '/' + aug_method[idx_data_aug - 1] + '_' + str(config['Augmentation_target_number']) + '_A_dict', A_dict)
            np.save('data/' + data_name + '/' + aug_method[idx_data_aug - 1] + '_' + str(config['Augmentation_target_number']) + '_trainQ_dict',
                    trainQ_dict)
            np.save('data/' + data_name + '/' + aug_method[idx_data_aug - 1] + '_' + str(config['Augmentation_target_number']) + '_testQ_dict',
                    testQ_dict)
    else:
        if idx_data_aug != 3 and os.path.exists('data/' + data_name + '/train_data.csv'):
            train_data = pd.read_csv('data/' + data_name + '/train_data.csv')
            train_data.columns = [0, 1, 'idQ', 'idA']
            test_data = pd.read_csv('data/' + data_name + '/test_data.csv')
            test_data.columns = [0, 1, 'idQ', 'idA']
            A_dict = eval(str(np.load('data/' + data_name + '/A_dict.npy',
                                      allow_pickle=True)))
            trainQ_dict = eval(
                str(np.load('data/' + data_name + '/trainQ_dict.npy',
                            allow_pickle=True)))
            testQ_dict = eval(str(np.load('data/' + data_name + '/testQ_dict.npy',
                                          allow_pickle=True)))
        elif idx_data_aug == 3 and os.path.exists('data/' + data_name + '/train_data.csv') and os.path.exists('data/' + data_name + '/imbalance_train.npy'):
            train_data = pd.read_csv('data/' + data_name + '/train_data.csv')
            train_data.columns = [0, 1, 'idQ', 'idA']
            test_data = pd.read_csv('data/' + data_name + '/test_data.csv')
            test_data.columns = [0, 1, 'idQ', 'idA']
            A_dict = eval(str(np.load('data/' + data_name + '/A_dict.npy', allow_pickle=True)))
            imbalance_train = eval(str(np.load('data/' + data_name + '/imbalance_train.npy', allow_pickle=True)))
            trainQ_dict = eval(str(np.load('data/' + data_name + '/trainQ_dict.npy', allow_pickle=True)))
            testQ_dict = eval(str(np.load('data/' + data_name + '/testQ_dict.npy', allow_pickle=True)))
        else:
            # train = pd.read_csv('data/' + data_name + '/train.tsv', sep='\t', header=None)
            # test = pd.read_csv('data/' + data_name + '/test.tsv', sep='\t', header=None)
            train = pd.read_csv('data/' + data_name + '/train.csv',header=None)
            test = pd.read_csv('data/' + data_name + '/test.csv',header=None)
            if idx_data_aug == 3:
                train_data, trainQ_dict, A_dict, imbalance_train = data_for_encoder(train, {}, implicit=True)
                np.save('data/' + data_name + '/imbalance_train', imbalance_train)
            else:
                train_data, trainQ_dict, A_dict = data_for_encoder(train, {})
            test_data, testQ_dict, A_dict = data_for_encoder(test, A_dict, False)
            train_data.to_csv('data/' + data_name + '/train_data.csv', index=False)
            test_data.to_csv('data/' + data_name + '/test_data.csv', index=False)
            np.save('data/' + data_name + '/A_dict', A_dict)
            np.save('data/' + data_name + '/trainQ_dict', trainQ_dict)
            np.save('data/' + data_name + '/testQ_dict', testQ_dict)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config['pretrain_model'])

    id2sentence_A = {v: k for k, v in A_dict.items()}

    # #######################################################
    # case_study_data = test_data.sample(n=20)
    # case_study_data.index = range(0, len(case_study_data))
    # ################################################
    if istest:
        print('Test begin:')
        model = naive_match_model(pretrain_model= config['pretrain_model'])
        PATH='checkpoints/'+config['dataset']+'/'+checkpoint
        print(PATH)
        model.load_state_dict(torch.load(PATH))
        model.eval()
        model = model.to(device)
        test_dataset = pretrainQ_Dataset(test_data, tokenizer)
        test_iter = data.DataLoader(
            dataset=test_dataset,
            batch_size=config['test_batch_size'],
            shuffle=False,
            num_workers=2)
        label_data = A_learn_dataset(list(A_dict.keys()), tokenizer)
        for label in label_data.keys():
            label_data[label] = label_data[label].to(device)
        label_feat = model.AEncoder(label_data)
        mask_test,_ = match_test(model, test_iter, label_feat, device, 1)

    else:
        print('Train begin:')
        acc1_list = []
        acc2_list = []
        f1_list = []
        for i in range(1):
            best_acc1 = 0
            best_acc2 = 0
            best_f1 = 0

            train_dataset = puretrain_dataset(train_data, tokenizer)
            train_data4mask = pretrainQ_Dataset(train_data, tokenizer)
            test_dataset = pretrainQ_Dataset(test_data, tokenizer)

            # case_study_data = pretrainQ_Dataset(case_study_data,tokenizer)
            # case_study_iter = data.DataLoader(
            # dataset=case_study_data,
            # batch_size=config['test_batch_size'],
            # shuffle=True,
            # num_workers=2)
        # #######################################################

            train_iter = data.DataLoader(
                dataset=train_dataset,
                batch_size=config['recall_train_batch_size'],
                shuffle=True,
                num_workers=2)
            train_iter4mask = data.DataLoader(
                dataset=train_data4mask,
                batch_size=config['recall_train_batch_size'],
                shuffle=False,
                num_workers=2)
            test_iter = data.DataLoader(
                dataset=test_dataset,
                batch_size=config['test_batch_size'],
                shuffle=False,
                num_workers=2)


            model = naive_match_model(pretrain_model= config['pretrain_model'])
            model = model.to(device)
            optimizer = optim.AdamW(model.parameters(), lr=config["learn_rate"])
            scheduler = StepLR(optimizer, step_size=config["lr_dc_step"], gamma=config["lr_dc"])

            # for test
            label_data = A_learn_dataset(list(A_dict.keys()), tokenizer)
            for label in label_data.keys():
                label_data[label] = label_data[label].to(device)
            print()
            print("Recall train start")
        #######################################################
            # or_cos,or_idx = case_study(model,case_study_iter,model.AEncoder(label_data),device)
            # cos_list1 = []
            # cos_list2 = []
            # #######################################################
            for epoch in range(1, config['train_epoch'] + 1):
                match_train(epoch, model, train_iter, imbalance_train, optimizer, device)
                scheduler.step(epoch=epoch)
                if epoch % 1 == 0:
                    print('Test result in epoch', epoch)
                    _,tempacc = match_test(model, test_iter, model.AEncoder(label_data), device, epoch)
                    # if tempacc > best_acc1:
                    #     cos_list1, _ = case_study(model, case_study_iter, model.AEncoder(label_data), device)
                    best_acc1 =  tempacc if tempacc > best_acc1 else best_acc1
            acc1_list.append(best_acc1)
            label_feat = model.AEncoder(label_data)
            print()
            print("Generating candidate label set for classification")
            print("Test Dataset result")
            mask_test,_ = match_test(model, test_iter, label_feat, device, epoch)
            print("Train Dataset result")
            mask_train,_ = match_test(model, train_iter4mask, label_feat, device, epoch)

            # Classification
            classes = [x for x in range(0, len(A_dict))]

            train_dataset = similarity_mask_dataset(train_data, range(len(mask_train)), tokenizer, istrain=True)
            test_dataset = similarity_mask_dataset(test_data, range(len(mask_test)), tokenizer, istrain=False)
            train_iter = data.DataLoader(
                dataset=train_dataset,
                batch_size=config['classification_train_batch_size'],
                shuffle=True,
                num_workers=2)
            test_iter = data.DataLoader(
                dataset=test_dataset,
                batch_size=config['test_batch_size'],
                shuffle=True,
                num_workers=2)
            optimizer = optim.AdamW(model.parameters(), lr=config["s2_learn_rate"])
            scheduler = StepLR(optimizer, step_size=config["s2_lr_dc_step"], gamma=config["s2_lr_dc"])
            print()
            print("Classification train start")
            max = 0
            for filename in glob.glob('data/'+config['dataset']+'/*.pth'):
                # print(filename)
                if max < float(filename.split('\\')[-1].split('_')[1]):
                    max = float(filename.split('\\')[-1].split('_')[1])
            max=0
            for epoch in range(1, config['s2_train_epoch']):
                similar_train(epoch, model, train_iter, mask_train, id2sentence_A, tokenizer, optimizer, device)
                scheduler.step(epoch=epoch)
                if epoch % 1 == 0:

                    print('Test result in epoch', epoch)
                    tempacc,tempf1 = similar_test(model, test_iter, mask_test, id2sentence_A, tokenizer, device)
                    # if tempacc > best_acc2:
                    #     cos_list2, __ = case_study(model, case_study_iter, model.AEncoder(label_data), device)
                    if tempacc > max:
                        max = tempacc
                        torch.save(model.state_dict(), 'data/'+config['dataset']+'/acc_' + str(tempacc) + '_f1_' + str(tempf1) + '.pth')
                    best_acc2 =  tempacc if tempacc > best_acc2 else best_acc2
                    best_f1 = tempf1 if tempf1 > best_f1 else best_f1
            # print()
            # print("Final test start")
            # similar_test(model, test_iter, mask_test, id2sentence_A, tokenizer, device)

            # acc2_list.append(best_acc2)
            # f1_list.append(best_f1)
            # print('acc1_list:',acc1_list)
            # print('acc2_list:',acc2_list)
            # print('f1_list:',f1_list)
            
            # print('orcos=',or_cos)
            # print('cos_list1=',cos_list1)
            # print('cos_list2=',cos_list2)
            # print('idx=',or_idx)





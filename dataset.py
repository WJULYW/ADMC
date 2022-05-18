import numpy as np
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from typing import Sequence, Optional
from configure.config import config


class puretrain_dataset(Dataset):
    def __init__(self, data, tokenizer):
        self.x = data[0]
        self.y = data[1]
        self.config = config
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        data_x = self.tokenizer.encode_plus(
            self.x[idx],
            add_special_tokens=True,
            max_length=self.config['max_seq_Q_len'],
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        data_x = {

            'input_ids': data_x['input_ids'].flatten(),
            'attention_mask': data_x['attention_mask'].flatten(),
            #'token_type_ids': data_x['token_type_ids'].flatten()

        }

        data_y = self.tokenizer.encode_plus(
            self.y[idx],
            add_special_tokens=True,
            max_length=self.config['max_seq_A_len'],
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        data_y = {
            'input_ids': data_y['input_ids'].flatten(),
            'attention_mask': data_y['attention_mask'].flatten(),
            #'token_type_ids': data_y['token_type_ids'].flatten()

        }

        return data_x, data_y, 1

    def __len__(self):
        return len(self.x)


class classification_mask_dataset(Dataset):
    def __init__(self, data, mask, tokenizer):
        self.x = data[0]
        self.a_id = data['idA']
        self.mask = mask
        self.config = config
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        data_x = self.tokenizer.encode_plus(
            self.x[idx],
            add_special_tokens=True,
            max_length=self.config['max_seq_Q_len'],
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        data_x = {
            'input_ids': data_x['input_ids'].flatten(),
            'attention_mask': data_x['attention_mask'].flatten(),
            'token_type_ids': data_x['token_type_ids'].flatten()

        }

        return data_x, self.mask[idx], self.a_id

    def __len__(self):
        return len(self.x)


class similarity_mask_dataset(Dataset):
    def __init__(self, data, range_mask, tokenizer, istrain=True):
        self.x = data[0]
        self.y = data[1]
        self.a_id = data['idA']
        self.range_mask = range_mask
        self.config = config
        self.tokenizer = tokenizer
        self.istrain = istrain

    def __getitem__(self, idx):
        data_x = self.tokenizer.encode_plus(
            self.x[idx],
            add_special_tokens=True,
            max_length=self.config['max_seq_Q_len'],
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        data_x = {
            'input_ids': data_x['input_ids'].flatten(),
            'attention_mask': data_x['attention_mask'].flatten(),
            'token_type_ids': data_x['token_type_ids'].flatten()

        }

        if self.istrain:
            data_y = self.tokenizer.encode_plus(
                self.y[idx],
                add_special_tokens=True,
                max_length=self.config['max_seq_A_len'],
                pad_to_max_length=True,
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            data_y = {
                'input_ids': data_y['input_ids'].flatten(),
                'attention_mask': data_y['attention_mask'].flatten(),
                'token_type_ids': data_y['token_type_ids'].flatten()

            }

            return data_x, data_y, self.range_mask[idx], self.a_id[idx]
        else:
            return data_x, self.range_mask[idx], self.a_id[idx]

    def __len__(self):
        return len(self.x)


class interaction_classification_Dataset(Dataset):
    def __init__(self, data, tokenizer):
        self.q = data[0]
        self.a = data[1]
        self.yid = data['idA']
        self.config = config
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        data_q = self.tokenizer.encode_plus(
            self.q[idx],
            add_special_tokens=True,
            max_length=self.config['max_seq_Q_len'],
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        data_q = {
            'input_ids': data_q['input_ids'].flatten(),
            'attention_mask': data_q['attention_mask'].flatten(),
            'token_type_ids': data_q['token_type_ids'].flatten()

        }
        data_a = self.tokenizer.encode_plus(
            self.a[idx],
            add_special_tokens=True,
            max_length=self.config['max_seq_A_len'],
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        data_a = {
            'input_ids': data_a['input_ids'].flatten(),
            'attention_mask': data_a['attention_mask'].flatten(),
            'token_type_ids': data_a['token_type_ids'].flatten()

        }
        return data_q, data_a, self.yid[idx]

    def __len__(self):
        return len(self.yid)


class pretrainQ_Dataset(Dataset):
    def __init__(self, data, tokenizer):
        self.x = data[0]
        self.yid = data['idA']
        ####################################
        # self.tc_1 = data['tc_0.01']
        # self.tc_5 = data['tc_0.05']
        # self.tc_10 = data['tc_0.1']
        # self.tc_15 = data['tc_0.15']
        ####################################
        self.config = config
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        data_x = self.tokenizer.encode_plus(
            self.x[idx],
            add_special_tokens=True,
            max_length=self.config['max_seq_Q_len'],
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        data_x = {
            'input_ids': data_x['input_ids'].flatten(),
            'attention_mask': data_x['attention_mask'].flatten(),
            'token_type_ids': data_x['token_type_ids'].flatten()

        }
        # return data_x, self.yid[idx],self.tc_1[idx],self.tc_5[idx],self.tc_10[idx],self.tc_15[idx]
        return data_x, self.yid[idx]

    def __len__(self):
        return len(self.yid)


class match_Dataset(Dataset):
    def __init__(self, data, label, tokenizer):
        self.q = data[0]
        self.a = data[1]
        self.y = label
        self.config = config
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        data_q = self.tokenizer.encode_plus(
            self.q[idx],
            add_special_tokens=True,
            max_length=self.config['max_seq_Q_len'],
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        data_q = {
            'input_ids': data_q['input_ids'].flatten(),
            'attention_mask': data_q['attention_mask'].flatten(),
            'token_type_ids': data_q['token_type_ids'].flatten()

        }
        data_a = self.tokenizer.encode_plus(
            self.a[idx],
            add_special_tokens=True,
            max_length=self.config['max_seq_A_len'],
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        data_a = {
            'input_ids': data_a['input_ids'].flatten(),
            'attention_mask': data_a['attention_mask'].flatten(),
            'token_type_ids': data_a['token_type_ids'].flatten()

        }
        return data_q, data_a, self.y[idx]

    def __len__(self):
        return len(self.y)


def A_learn_dataset(data, tokenizer):
    data = tokenizer(
        data,
        add_special_tokens=True,
        max_length=config['max_seq_Q_len'],
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    '''
    data = {
        'input_ids': data['input_ids'].flatten(),
        'attention_mask': data['attention_mask'].flatten(),
        'token_type_ids': data['token_type_ids'].flatten()
    }'''
    return data

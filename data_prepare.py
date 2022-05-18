import random
import pandas as pd
from google_trans_new import google_translator
from configure.config import config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def find_tail_class(train, test,tail_pacentage,A_dict):
    count_df = train.groupby([1], as_index=False)[1].agg({'cnt': 'count'})
    number_of_class = len(count_df)
    count_df = count_df.sort_values('cnt')
    count_df.index = range(0, number_of_class)
    threshold_ = round(number_of_class*tail_pacentage)
    threshold_ = 1 if threshold_ == 0 else threshold_
    threshold = int(count_df[threshold_-1:threshold_]['cnt'])
    count_df = list(count_df[count_df['cnt'] <= threshold][1])
    tail_class = []
    for i in count_df:
        tail_class.append(A_dict[i])
    test['tc_'+str(tail_pacentage)] = test['idA'].apply(lambda x: True if x in tail_class else False)
    # print(test[test['tc_0.01'] == True][1])
    return test

def func_for_low_Paraphrase(dataset,groups,target,tokenizer,model,device):
    new_x = []
    new_y = []
    for i in groups.keys():
        y = i
        str = groups[i][0]
        end = groups[i][-1]
        cnt = int(dataset[str:str+1]['cnt'])
        needed_generat = target - cnt
        per_data = needed_generat//cnt
        temp = dataset[str:end+1]
        temp = list(temp[0])
        generated_count = 0
        for j in temp:
            encoding = tokenizer.encode_plus(j, pad_to_max_length=True, return_tensors="pt")
            input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
            if j != temp[-1]:
                outputs = model.generate(
                    input_ids=input_ids, attention_mask=attention_masks,
                    max_length=64,
                    do_sample=True,
                    top_k=120,
                    top_p=0.95,
                    early_stopping=True,
                    num_return_sequences=per_data
                )
                generated_count += per_data
            else:
                outputs = model.generate(
                    input_ids=input_ids, attention_mask=attention_masks,
                    max_length=64,
                    do_sample=True,
                    top_k=120,
                    top_p=0.95,
                    early_stopping=True,
                    num_return_sequences= needed_generat - generated_count
                )
            for output in outputs:
                line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                new_x.append(line)
                new_y.append(y)
    return pd.DataFrame({0:new_x,1:new_y})

def func_for_heigh_Paraphrase(data,tokenizer,model,device):
    encoding = tokenizer.encode_plus(data, pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=64,
        do_sample=True,
        top_k=120,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=1
    )
    line = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return line

def data_augmentation_Paraphrase(dataset,target): #max:429
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    model.to(device)

    aug_data = dataset.groupby([1], as_index=False)[1].agg({'cnt': 'count'})

    aug_data = pd.merge(dataset, aug_data, how='inner', on=[1])

    aug_data_low = aug_data[aug_data['cnt'] <= target // 2]

    if len(aug_data_low) > 0:
        aug_data_low.index = range(0, len(aug_data_low))
        aug_data_low[0] = aug_data_low[0].apply(lambda x:"paraphrase: " + x + " </s>")
        grouped = aug_data_low.groupby(1, group_keys=True).groups
        aug_data_low = func_for_low_Paraphrase(aug_data_low,grouped,target,tokenizer,model,device)
        if len(aug_data_low) > 0:
            dataset = pd.concat([dataset, aug_data_low])


    aug_data_heigh = aug_data[(aug_data['cnt'] > target // 2) & (aug_data['cnt'] < target)]
    if len(aug_data_heigh) > 0:
        grouped = aug_data_heigh.groupby(1, group_keys=False)
        aug_data_heigh = grouped.apply(lambda x: x.sample(target-x['cnt'].iloc[0]))
        aug_data_heigh[0] = aug_data_heigh[0].apply(lambda x:"paraphrase: " + x + " </s>")
        aug_data_heigh[0] = aug_data_heigh[0].apply(lambda x: func_for_heigh_Paraphrase(x,tokenizer,model,device))

    if len(aug_data_heigh) > 0:
        aug_data_heigh = aug_data_heigh.drop('cnt', axis=1)
        dataset = pd.concat([dataset, aug_data_heigh])

    dataset.index = range(0, len(dataset))
    return dataset


def pre_translation(aug_data, target,translator):
    lang_list = ['af', 'sq', 'az', 'eu', 'be', 'bn', 'bs', 'bg', 'ca', 'ceb', 'ny', 'zh-CN', 'zh-TW', 'co', 'cs', 'da',
                 'nl', 'eo', 'et', 'fi', 'fr', 'fy', 'gl', 'ka', 'de', 'el', 'gu', 'ht', 'ha', 'haw', 'hi', 'hmn', 'hu',
                 'is', 'ig', 'id', 'ga', 'jw', 'kn', 'kk', 'km', 'ku', 'lo', 'la', 'lv', 'lt', 'lb', 'mk', 'mg', 'ms',
                 'ml', 'mt', 'mi', 'mr', 'mn', 'ne', 'no', 'or', 'pl', 'pt', 'sm', 'sr', 'st', 'sn', 'sd', 'si', 'sk',
                 'sl', 'su', 'sw', 'sv', 'tg', 'ta', 'te', 'th', 'tr', 'uk', 'ur', 'vi', 'cy', 'yi', 'zu']
    lang = random.choice(lang_list)
    aug_data_low = aug_data[aug_data['cnt'] <= target//2]
    if len(aug_data_low) > 0 :
        aug_data_low[0] = aug_data_low[0].apply(lambda x:
            translator.translate(translator.translate(x, lang_src='en', lang_tgt=lang), lang_src=lang, lang_tgt='en'))

    aug_data_heigh = aug_data[aug_data['cnt'] > target//2]
    if len(aug_data_heigh) > 0:
        grouped = aug_data_heigh.groupby(1, group_keys=False)
        aug_data_heigh = grouped.apply(lambda x: x.sample(target-x['cnt'].iloc[0]))
        aug_data_heigh[0] = aug_data_heigh[0].apply(lambda x:
            translator.translate(translator.translate(x, lang_src='en', lang_tgt=lang), lang_src=lang, lang_tgt='en'))

    return aug_data_low, aug_data_heigh


def data_augmentation_translation(dataset,target): #max:429
    translator = google_translator()  #new

    aug_data = dataset.groupby([1], as_index=False)[1].agg({'cnt': 'count'})

    aug_data = pd.merge(dataset, aug_data, how='inner', on=[1])
    aug_data = aug_data[aug_data['cnt'] < target]
    while len(aug_data) > 0:
        aug_data_low, aug_data_heigh = pre_translation(aug_data, target,translator)

        if len(aug_data_low) > 0:
            aug_data_low = aug_data_low.drop('cnt', axis=1)
            dataset = pd.concat([dataset, aug_data_low])
        if len(aug_data_heigh) > 0:
            aug_data_heigh = aug_data_heigh.drop('cnt', axis=1)
            dataset = pd.concat([dataset, aug_data_heigh])

        aug_data = dataset.groupby([1], as_index=False)[1].agg({'cnt': 'count'})
        aug_data = pd.merge(dataset, aug_data, how='inner', on=[1])
        aug_data = aug_data[aug_data['cnt'] < target]

    dataset.index = range(0, len(dataset))
    return dataset

def produce_sentence2id(Q, A, istrain=True):
    Q_set = set(Q)
    A_set = set(A)
    sentence2idQ = dict()
    for key, value in enumerate(Q_set):
        sentence2idQ[value] = key
    if istrain:
        sentence2idA = dict()
        for key, value in enumerate(A_set):
            sentence2idA[value] = key
        return sentence2idQ, sentence2idA
    else:
        return sentence2idQ


def produce_Q_positive_group(data, sentence2idQ, sentence2idA):
    pos_group = dict()
    for i, j in zip(data[0], data[1]):
        j = sentence2idA[j]
        i = sentence2idQ[i]
        if j not in pos_group.keys():
            pos_group[j] = [i]
        else:
            pos_group[j].append(i)
    return pos_group


def statistic_imbalance(data):
    res = dict()
    sta = dict()
    for line in data.values:
        if line[3] in sta.keys():
            sta[line[3]] += 1
        else:
            sta[line[3]] = 1
    for k in sta.keys():
        if sta[k] < config['Augmentation_target_number']:
            res[k] = (config['Augmentation_target_number'] - sta[k]) // sta[k]
    return res


def data_for_encoder(data, sentence2idA, istrain=True, implicit=False):
    if istrain:
        sentence2idQ, sentence2idA = produce_sentence2id(data[0], data[1], istrain)
    else:
        sentence2idQ = produce_sentence2id(data[0], data[1], istrain)
    idQ = [sentence2idQ[v] for v in data[0]]
    data['idQ'] = idQ
    idA = [sentence2idA[v] for v in data[1]]
    data['idA'] = idA
    if implicit:
        imbalance = statistic_imbalance(data)
        return data, sentence2idQ, sentence2idA, imbalance
    return data, sentence2idQ, sentence2idA


def data_for_neg(data, sentence2idQ, sentence2idA):
    res = []
    new_label = []
    data_c = data.to_numpy().tolist()
    for record in data_c:
        res.append(record)
        for n in random.sample(set(list(sentence2idA.keys())) - set([record[1]]), config["negative_label_num"]):
            temp = record[:]
            temp[1] = n
            temp[3] = sentence2idA[n]
            res.append(temp)

        new_label.append(1)
        new_label += [0 for i in range(config["negative_label_num"])]
    return pd.DataFrame(res, columns=data.columns), new_label


def dict4contrastive(data, sentence2idQ):
    pos = {}
    neg = {}
    data_c = data.to_numpy().tolist()
    for record in data_c:
        if record[3] not in pos.keys():
            pos[record[3]] = [record[0]]
        else:
            pos[record[3]].append(record[0])
        neg[record[3]] = random.sample(set(list(sentence2idQ.keys())) - set(pos[record[3]]), config["negative_label_num"])
    fusion = []
    for i in list(pos.keys()):
        fusion.append([])
        for j in list(pos.keys()):
            if i == j:
                fusion[-1].append(1)
            else:
                fusion[-1].append(similar_score(pos[i], pos[j]))

    return pos, neg, fusion


def similar_score(l1, l2):
     return sum([1 if i in l2 else 0 for i in l1]) / len(l1)

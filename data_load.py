import pandas as pd
import numpy as np
import json
import os
import random
import kss

import torch
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings(action='ignore')


DIR = "./data"

TRAIN_SOURCE = os.path.join(DIR, "train.json")
TEST_SOURCE = os.path.join(DIR, "test.json")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def _pad(data, pad_id, width=-1):
    if (width == -1):
        width = max(len(d) for d in data)

    # test
    if any([width - len(d) < 0 for d in data]):
        ffsfadsfa

    rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
    return rtn_data


def evidence_preprocess(dataframe):
    # string -> list
    for i in range(len(dataframe)):
        if 'sentence' in dataframe.keys() and isinstance(dataframe.at[i, 'sentence'], str):
            dataframe.at[i, 'sentence'] = dataframe.at[i, 'sentence'][2:-2].split("', '")
        
        if 'evidence' in dataframe.keys() and isinstance(dataframe.at[i, 'evidence'], str):
            dataframe.at[i, 'evidence'] = dataframe.at[i, 'evidence'][2:-2].split("', '")

    # preprocessing
    for i in range(len(dataframe)):
        if dataframe.at[i, 'sentence'] == [] or not 'evidence' in dataframe.keys():
            break
        if not all([ev in dataframe.at[i, 'sentence'] for ev in dataframe.at[i, 'evidence']]):
            dataframe.at[i, 'evidence'] = [temp.replace('  ', ' ') for temp in dataframe.at[i, 'evidence']]

            new_evi = []
            for ev_i in range(len(dataframe.at[i, 'evidence'])):
                if any(dataframe.at[i, 'evidence'][ev_i] == temp for temp in dataframe.at[i, 'sentence']):
                    new_evi.append(dataframe.at[i, 'evidence'][ev_i])
                    continue
                else:
                    for temp in dataframe.at[i, 'sentence']:
                        if dataframe.at[i, 'evidence'][ev_i] in temp and '의원 여러분!' in dataframe.at[i, 'evidence'][ev_i] and '없으십니까?' in dataframe.at[i, 'evidence'][ev_i+1]:
                            new_evi.append(temp)
                            # print((' ').join(dataframe.at[i, 'evidence'][ev_i:ev_i+2]) == temp)
                            break
                        elif dataframe.at[i, 'evidence'][ev_i] in temp and '의원 여러분!' in dataframe.at[i, 'evidence'][ev_i-1] and '없으십니까?' in dataframe.at[i, 'evidence'][ev_i]:
                            continue
            if all([any(newev == temp for temp in dataframe.at[i, 'sentence']) for newev in new_evi]):
                dataframe.at[i, 'evidence'] = new_evi
            else:
                print(i)
                print(dataframe.at[i, 'sentence'])
                print(dataframe.at[i, 'evidence'])
                print('new_evi')
                print(len(new_evi), new_evi)
                print('')
        # break
    return dataframe


def data_loader():
    if os.path.isfile('./data/train_all.csv') and os.path.isfile('./data/val_all.csv') and os.path.isfile('./data/test_all.csv'):
        print('load train_all.csv..')
        df_train = pd.read_csv("./data/train_all.csv")
        df_val = pd.read_csv("./data/val_all.csv")
        test = pd.read_csv("./data/test_all.csv")
        print(len(df_train), len(df_val), len(test))

        df_train = evidence_preprocess(df_train)
        df_val = evidence_preprocess(df_val)
        test = evidence_preprocess(test)

        return df_train, df_val, test

    with open(TRAIN_SOURCE) as f:
        TRAIN_DATA = json.loads(f.read())

    with open(TEST_SOURCE) as f:
        TEST_DATA = json.loads(f.read())

    train = pd.DataFrame(columns=['uid', 'title', 'region', 'context', 'evidence', 'sentence', 'summary'])
    uid = 1000
    for data in TRAIN_DATA:
        for agenda in data['context'].keys():
            context = ''
            sentence = []
            for line in data['context'][agenda]:
                context += data['context'][agenda][line]
                context += ' '

                # (『없습니다』하는 의원 있음)
                if len(data['context'][agenda][line]) >= 39 and len(data['context'][agenda][line]) <= 43 and all(i in data['context'][agenda][line] for i in ['없습니다', '하는', '의원', '있음']):
                    sentence += [data['context'][agenda][line]]
                else:
                    # '(의석에서)  솰라솰라'
                    if '(의석에서)' in data['context'][agenda][line]:
                        sentence += kss.split_sentences(data['context'][agenda][line].replace('\u2003', ''))
                    else:
                        try:
                            sentence += kss.split_sentences(data['context'][agenda][line])
                        except:
                            sentence += [data['context'][agenda][line]]
                            print(data['context'][agenda][line])
                            #print(kss.split_sentences(data['context'][agenda][line]))

            evidence = []
            for line in data['label'][agenda]['evidence']:
                evidence += data['label'][agenda]['evidence'][line]

            train.loc[uid, 'uid'] = uid
            train.loc[uid, 'title'] = data['title']
            train.loc[uid, 'region'] = data['region']
            train.loc[uid, 'context'] = context[:-1]
            train.loc[uid, 'evidence'] = evidence
            train.loc[uid, 'sentence'] = sentence
            train.loc[uid, 'summary'] = data['label'][agenda]['summary']
            uid += 1#;print(context)

    test = pd.DataFrame(columns=['uid', 'title', 'region', 'context'])
    uid = 2000
    for data in TEST_DATA:
        for agenda in data['context'].keys():
            context = ''
            sentence = []
            for line in data['context'][agenda]:
                context += data['context'][agenda][line]
                context += ' '

                # (『없습니다』하는 의원 있음)
                if len(data['context'][agenda][line]) >= 39 and len(data['context'][agenda][line]) <= 43 and all(i in data['context'][agenda][line] for i in ['없습니다', '하는', '의원', '있>음']):
                    sentence += [data['context'][agenda][line]]
                else:
                    # '(의석에서)  솰라솰라'
                    if '(의석에서)' in data['context'][agenda][line]:
                        sentence += kss.split_sentences(data['context'][agenda][line].replace('\u2003', ''))
                    else:
                        try:
                            sentence += kss.split_sentences(data['context'][agenda][line])
                        except:
                            sentence += [data['context'][agenda][line]]
                            print(data['context'][agenda][line])
                            #print(kss.split_sentences(data['context'][agenda][line]))


            test.loc[uid, 'uid'] = uid
            test.loc[uid, 'title'] = data['title']
            test.loc[uid, 'region'] = data['region']
            test.loc[uid, 'context'] = context[:-1]
            test.loc[uid, 'sentence'] = sentence
            uid += 1

    train['total'] = train.title + ' ' + train.region + ' ' + train.context
    test['total'] = test.title + ' ' + test.region + ' ' + test.context

    df_train = train.iloc[:-200]
    df_val = train.iloc[-200:]

    if not os.path.isfile('./data/train_all.csv') or not os.path.isfile('./data/val_all.csv') or not os.path.isfile('./data/test_all.csv'):
        df_train.to_csv('data/train_all.csv')
        df_val.to_csv('data/val_all.csv')
        test.to_csv('data/test_all.csv')
        print(len(df_train), len(df_val), len(test))

    return df_train, df_val, test


class CustomDataset(Dataset):
    def __init__(self, src_tokens, tar_tokens, src_segs, clss, src_sent_labels, mode='train'):
        self.mode = mode
        self.src_tokens = src_tokens
        self.src_masks = 1 - (src_tokens == 0)
        self.src_segs = src_segs
        self.cls = clss
        self.mask_cls = 1 - (clss == -1)
        self.cls[self.cls == -1] = 0
        self.src_sent_labels = src_sent_labels

        if self.mode == 'train':
            self.tar_tokens = tar_tokens
            self.tar_masks = 1 - (tar_tokens == 0)

    def __len__(self):
        return len(self.src_tokens)

    def __getitem__(self, i):
        src_token = self.src_tokens[i]
        src_mask = self.src_masks[i]
        src_seg = self.src_segs[i]
        cls = self.cls[i]
        mask_cls = self.mask_cls[i]
        src_sent_labels = self.src_sent_labels[i]
        if self.mode == 'train':
            tar_token = self.tar_tokens[i]
            tar_mask = self.tar_masks[i]
            return {
                'src_token' : torch.tensor(src_token, dtype=torch.long),
                'tar_token' : torch.tensor(tar_token, dtype=torch.long),
                'src_mask'  : torch.tensor(src_mask, dtype=torch.long),
                'tar_mask'  : torch.tensor(tar_mask, dtype=torch.long),
                'src_seg'   : torch.tensor(src_seg, dtype=torch.long),
                'clss'      : torch.tensor(cls, dtype=torch.long),
                'mask_cls'  : torch.tensor(mask_cls, dtype=torch.bool),
                'src_sent_labels'  : torch.tensor(src_sent_labels, dtype=torch.long),
            }
        else:
            return {
                'src_token' : torch.tensor(src_token, dtype=torch.long),
                'src_mask'  : torch.tensor(src_mask, dtype=torch.long),
                'src_seg'   : torch.tensor(src_seg, dtype=torch.long),
                'clss'      : torch.tensor(cls, dtype=torch.long),
                'mask_cls'  : torch.tensor(mask_cls, dtype=torch.long),
                'src_sent_labels'  : torch.tensor(src_sent_labels, dtype=torch.long),
            }

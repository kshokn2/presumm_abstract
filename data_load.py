import pandas as pd
import numpy as np
import json
import os
import random

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


def data_loader():
    with open(TRAIN_SOURCE) as f:
        TRAIN_DATA = json.loads(f.read())

    with open(TEST_SOURCE) as f:
        TEST_DATA = json.loads(f.read())

    train = pd.DataFrame(columns=['uid', 'title', 'region', 'context', 'summary'])
    uid = 1000
    for data in TRAIN_DATA:
        for agenda in data['context'].keys():
            context = ''
            for line in data['context'][agenda]:
                context += data['context'][agenda][line]
                context += ' '
            train.loc[uid, 'uid'] = uid
            train.loc[uid, 'title'] = data['title']
            train.loc[uid, 'region'] = data['region']
            train.loc[uid, 'context'] = context[:-1]
            train.loc[uid, 'summary'] = data['label'][agenda]['summary']
            uid += 1

    test = pd.DataFrame(columns=['uid', 'title', 'region', 'context'])
    uid = 2000
    for data in TEST_DATA:
        for agenda in data['context'].keys():
            context = ''
            for line in data['context'][agenda]:
                context += data['context'][agenda][line]
                context += ' '
            test.loc[uid, 'uid'] = uid
            test.loc[uid, 'title'] = data['title']
            test.loc[uid, 'region'] = data['region']
            test.loc[uid, 'context'] = context[:-1]
            uid += 1

    train['total'] = train.title + ' ' + train.region + ' ' + train.context
    test['total'] = test.title + ' ' + test.region + ' ' + test.context

    df_train = train.iloc[:-200]
    df_val = train.iloc[-200:]

    return df_train, df_val, test


class CustomDataset(Dataset):
    def __init__(self, src_tokens, tar_tokens, src_segs, mode='train'):
        self.mode = mode
        self.src_tokens = src_tokens
        self.src_masks = 1 - (src_tokens == 1)
        self.src_segs = src_segs
        if self.mode == 'train':
            self.tar_tokens = tar_tokens
            self.tar_masks = 1 - (tar_tokens == 1)

    def __len__(self):
        return len(self.src_tokens)

    def __getitem__(self, i):
        src_token = self.src_tokens[i]
        src_mask = self.src_masks[i]
        src_seg = self.src_segs[i]
        if self.mode == 'train':
            tar_token = self.tar_tokens[i]
            tar_mask = self.tar_masks[i]
            return {
                'src_token' : torch.tensor(src_token, dtype=torch.long),
                'tar_token' : torch.tensor(tar_token, dtype=torch.long),
                'src_mask'  : torch.tensor(src_mask, dtype=torch.long),
                'src_seg'   : torch.tensor(src_seg, dtype=torch.long),
                'tar_mask'  : torch.tensor(tar_mask, dtype=torch.long),
            }
        else:
            return {
                'src_token' : torch.tensor(src_token, dtype=torch.long),
                'src_mask'  : torch.tensor(src_mask, dtype=torch.long),
                'src_seg'   : torch.tensor(src_seg, dtype=torch.long),
            }

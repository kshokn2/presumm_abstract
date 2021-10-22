#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import json
import os
import argparse
#import math
#import copy
import gc
from tqdm import tqdm
#from glob import glob
#from konlpy.tag import Mecab

from data_load import seed_everything, CustomDataset, data_loader
from tokenizer import Mecab_Tokenizer
import params

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import xavier_uniform_
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertConfig
from models.absract import AbsSummarizer

import warnings
warnings.filterwarnings(action='ignore')


hparams = params.HParams()
device = hparams.device


def _save(args, step, real_model, loss_plot, acc_plot, val_loss_plot, val_acc_plot):
    model_state_dict = real_model.state_dict()

    checkpoint = {
        'model': model_state_dict,
        'opt': args,
        'loss_plot': loss_plot,
        'val_loss_plot': val_loss_plot,
        'acc_plot': acc_plot,
        'val_acc_plot': val_acc_plot,
    }
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    checkpoint_path = os.path.join(args.save_dir, 'model_step_%d.pt' % step)
    
    if not os.path.exists(checkpoint_path):
        print("Saving checkpoint %s" % checkpoint_path)
        torch.save(checkpoint, checkpoint_path)


def main(args):
    hparams = params.HParams()

    # load checkpoint with args
    if args.checkpoint != "":
        checkpoint_ = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)
        opt = vars(checkpoint_['opt'])
        for k in opt.keys():
            setattr(args, k, opt[k])
    else:
        checkpoint_ = None

    # data load
    df_train, df_val, test = data_loader()

    # tokenizer
    src_tokenizer = Mecab_Tokenizer(hparams.encoder_len, mode='enc', max_vocab_size=hparams.max_vocab_size)
    tar_tokenizer = Mecab_Tokenizer(hparams.decoder_len, mode='dec', max_vocab_size=hparams.max_vocab_size)

    train_src = src_tokenizer.morpheme(df_train.total)
    val_src = src_tokenizer.morpheme(df_val.total)
    test_src = src_tokenizer.morpheme(test.total)

    train_tar = tar_tokenizer.morpheme(df_train.summary)
    val_tar = tar_tokenizer.morpheme(df_val.summary)

    # original
    #src_tokenizer.fit(train_src)
    #tar_tokenizer.fit(train_tar)

    # tokenizer test1
    src_tokenizer.fit(train_src + val_src + test_src + train_tar + val_tar)
    tar_tokenizer.fit(train_src + val_src + test_src + train_tar + val_tar)

    # tokenizer test2
    #src_tokenizer.fit(train_src + val_src + test_src)
    #tar_tokenizer.fit(train_tar + val_tar)

    train_src_tokens = src_tokenizer.txt2token(train_src)
    val_src_tokens = src_tokenizer.txt2token(val_src)
    test_src_tokens = src_tokenizer.txt2token(test_src)

    train_tar_tokens = tar_tokenizer.txt2token(train_tar)
    val_tar_tokens = tar_tokenizer.txt2token(val_tar)

    train_src_segs = src_tokenizer.token2seg(train_src_tokens)
    val_src_segs = src_tokenizer.token2seg(val_src_tokens)

    args.input_vocab_size = len(src_tokenizer.txt2idx)
    args.target_vocab_size = len(tar_tokenizer.txt2idx)

    train_dataset = CustomDataset(train_src_tokens, train_tar_tokens, train_src_segs)
    val_dataset = CustomDataset(val_src_tokens, val_tar_tokens, val_src_segs)
    test_dataset = CustomDataset(test_src_tokens, None, 'test')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.batch_size, num_workers=1, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=hparams.batch_size, num_workers=1, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=hparams.batch_size, num_workers=1, shuffle=False)

    # model
    model = AbsSummarizer(args, hparams.device, checkpoint=checkpoint_, bert_from_extractive=None)
    #print(model)


    def train_step(batch_item, epoch, batch, training):
        src = batch_item['src_token'].to(device)
        tar = batch_item['tar_token'].to(device)
        segs = batch_item['src_seg'].to(device)
        mask_src = batch_item['src_mask'].to(device)
        mask_tgt = batch_item['tar_mask'].to(device)

        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        if training is True:
            model.train()
            optimizer.zero_grad()
            #model.zero_grad()
            with torch.cuda.amp.autocast():
                output, _ = model(src, tar, segs, None, mask_src, mask_tgt, None)
                # output, _, _ = transformer([src, tar_inp, None])
                loss = loss_function(tar_real, output)
            acc = accuracy_function(tar_real, output)
            loss.backward()
            optimizer.step()
            lr = optimizer.param_groups[0]["lr"]
            return loss, acc, round(lr, 10)
        else:
            model.eval()
            with torch.no_grad():
                output, _ = model(src, tar, segs, None, mask_src, mask_tgt, None)
                # output, _, _ = transformer([src, tar_inp, None])
                loss = loss_function(tar_real, output)
            acc = accuracy_function(tar_real, output)
            return loss, acc, 0


    # opt, loss func
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate)
    criterion = nn.CrossEntropyLoss()

    def loss_function(real, pred):
        mask = torch.logical_not(torch.eq(real, 0))
        loss_ = criterion(pred.permute(0,2,1), real)
        mask = torch.tensor(mask, dtype=loss_.dtype)
        loss_ = mask * loss_

        return torch.sum(loss_)/torch.sum(mask)


    def accuracy_function(real, pred):
        accuracies = torch.eq(real, torch.argmax(pred, dim=2))
        mask = torch.logical_not(torch.eq(real, 0))
        accuracies = torch.logical_and(mask, accuracies)
        accuracies = torch.tensor(accuracies, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return torch.sum(accuracies)/torch.sum(mask)


    # training
    loss_plot, val_loss_plot = [], []
    acc_plot, val_acc_plot = [], []
    # epochs = 1

    for epoch in range(hparams.epochs):
        gc.collect()
        total_loss, total_val_loss = 0, 0
        total_acc, total_val_acc = 0, 0

        tqdm_dataset = tqdm(enumerate(train_dataloader))
        training = True
        for batch, batch_item in tqdm_dataset:
            batch_loss, batch_acc, lr = train_step(batch_item, epoch, batch, training)
            total_loss += batch_loss
            total_acc += batch_acc
            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'LR' : lr,
                'Loss': '{:06f}'.format(batch_loss.item()),
                'Total Loss' : '{:06f}'.format(total_loss/(batch+1)),
                'Total ACC' : '{:06f}'.format(total_acc/(batch+1))
            });#break # break test
        #break # break test
        loss_plot.append(total_loss/(batch+1))
        acc_plot.append(total_acc/(batch+1))
        
        tqdm_dataset = tqdm(enumerate(val_dataloader))
        training = False
        for batch, batch_item in tqdm_dataset:
            batch_loss, batch_acc, _ = train_step(batch_item, epoch, batch, training)
            total_val_loss += batch_loss
            total_val_acc += batch_acc
            
            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'Val Loss': '{:06f}'.format(batch_loss.item()),
                'Total Val Loss' : '{:06f}'.format(total_val_loss/(batch+1)),
                'Total Val ACC' : '{:06f}'.format(total_val_acc/(batch+1))
            })
        val_loss_plot.append(total_val_loss/(batch+1))
        val_acc_plot.append(total_val_acc/(batch+1))

        if ((epoch+1) % args.save_checkpoint_steps == 0) and epoch > 0:
            _save(args, epoch+1, model, loss_plot, acc_plot, val_loss_plot, val_acc_plot)
        #break


if __name__ == '__main__':
    seed_everything(42)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-save_dir", default="./checkpoint", type=str)
    parser.add_argument("-checkpoint", default="", type=str)
    args = parser.parse_args()

    args.visible_gpus = '-1'
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.save_checkpoint_steps = 5

    args.temp_dir = './temp_dir'
    args.finetune_bert = True
    args.max_pos = hparams.d_model #512
    args.share_emb = False
    args.use_bert_emb = False #True or False # False 해놓은 이유= baseline의 tgt vocab과 src vocab수가 다름.. 왜??
    args.sep_optim = True # seperate optimizer (encoder, decoder)
    args.accum_count = 1

    # encoder
    args.enc_dropout = 0.2
    args.enc_layers = 6 #6
    args.enc_hidden_size = 512
    args.enc_ff_size = hparams.dff #dff or 512
    # decoder
    args.dec_dropout = 0.2
    args.dec_layers = 6
    args.dec_hidden_size = 512 #512 or 768
    args.dec_heads = 8
    args.dec_ff_size = hparams.dff #2048

    args.param_init = 0
    args.param_init_glorot = True
    args.optim = 'adam'
    args.lr = 0.05
    args.lr_bert = 2e-3
    args.lr_dec = 2e-3
    args.beta1 = 0.9
    args.beta2 = 0.999
    args.warmup_steps = 8000
    args.warmup_steps_bert = 8000
    args.warmup_steps_dec = 8000
    args.max_grad_norm = 0

    main(args)
    print('done')

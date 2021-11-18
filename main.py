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

from data_load import seed_everything, _pad, CustomDataset, data_loader
from tokenizer import Mecab_Tokenizer, KoBertTokenizer
import params

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import xavier_uniform_
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertConfig
from models.absract import AbsSummarizer
from models.extract import ExtSummarizer
from models.transformer import ExtTransformer

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
    BiCrossEntropy = torch.nn.BCELoss(reduction='none')

    # load checkpoint with args
    if args.checkpoint != "":
        checkpoint_ = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
        opt = vars(checkpoint_['opt'])
        for k in opt.keys():
            setattr(args, k, opt[k])
    else:
        checkpoint_ = None

    # data load
    print('data loading..')
    df_train, df_val, test = data_loader()

    '''
    # test
    src_tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
    print(len(src_tokenizer.token2idx.keys()))
    for sents in df_train.sentence:
        print(sents, len(sents))
        print('[CLS] '+sents[0]+' [SEP]')
        print([ src_tokenizer.tokenize('[CLS] '+sents[i]+' [SEP]') for i in range(len(sents)) ])
        break
    '''

    # tokenizer
    src_tokenizer = Mecab_Tokenizer(hparams.encoder_len, mode='enc', max_vocab_size=hparams.max_vocab_size)
    tar_tokenizer = Mecab_Tokenizer(hparams.decoder_len, mode='dec', max_vocab_size=hparams.max_vocab_size)

    # src
    train_src = src_tokenizer.morpheme(df_train.sentence) # df_train.total
    val_src = src_tokenizer.morpheme(df_val.sentence) # df_val.total
    test_src = src_tokenizer.morpheme(test.sentence) # test.total

    # clss
    train_clss, len_train_clss = src_tokenizer.sent2clss(train_src)
    val_clss, len_val_clss = src_tokenizer.sent2clss(val_src)
    args.max_sentnum = max(len_train_clss + len_val_clss)

    # tar
    train_tar = tar_tokenizer.morpheme(df_train.summary)
    val_tar = tar_tokenizer.morpheme(df_val.summary)

    # original
    #src_tokenizer.fit(train_src)
    #tar_tokenizer.fit(train_tar)

    # tokenizer test1
    src_tokenizer.fit(train_src + val_src + test_src + train_tar + val_tar) # + train_evi + val_evi
    tar_tokenizer.txt2idx = src_tokenizer.txt2idx
    tar_tokenizer.idx2txt = src_tokenizer.idx2txt
    #tar_tokenizer.fit(train_src + val_src + test_src + train_tar + val_tar)

    # tokenizer test2
    #src_tokenizer.fit(train_src + val_src + test_src)
    #tar_tokenizer.fit(train_tar + val_tar)

    # src token
    train_src_tokens = src_tokenizer.txt2token(train_src)
    val_src_tokens = src_tokenizer.txt2token(val_src)
    #test_src_tokens = src_tokenizer.txt2token(test_src)

    # tar token
    train_tar_tokens = tar_tokenizer.txt2token(train_tar)
    val_tar_tokens = tar_tokenizer.txt2token(val_tar)

    # segs
    train_src_segs = src_tokenizer.token2seg(train_src_tokens)
    val_src_segs = src_tokenizer.token2seg(val_src_tokens)

    args.input_vocab_size = len(src_tokenizer.txt2idx)
    args.target_vocab_size = len(tar_tokenizer.txt2idx)

    # labels
    train_src_sent_labels = tar_tokenizer.sent_label(df_train.sentence, df_train.evidence, args.max_sentnum)
    val_src_sent_labels = tar_tokenizer.sent_label(df_val.sentence, df_val.evidence, args.max_sentnum)

    # padding...
    train_clss = np.array(_pad(train_clss, -1, args.max_sentnum))
    val_clss = np.array(_pad(val_clss, -1, args.max_sentnum))
    train_src_sent_labels = np.array(_pad(train_src_sent_labels, 0, args.max_sentnum))
    val_src_sent_labels = np.array(_pad(val_src_sent_labels, 0, args.max_sentnum))
    all_neg_labels=0
    print(len(train_src_sent_labels))
    print('all negative class case of sentence', len([i for i, labels_test in enumerate(train_src_sent_labels) if sum(labels_test) == 0]))

    #print(train_src_tokens.shape);print(train_tar_tokens.shape);print(train_src_segs.shape);print(train_clss.shape);print(train_src_sent_labels.shape);print(train_src_sent_labels[0])
    train_dataset = CustomDataset(train_src_tokens, train_tar_tokens, train_src_segs, train_clss, train_src_sent_labels)
    val_dataset = CustomDataset(val_src_tokens, val_tar_tokens, val_src_segs, val_clss, val_src_sent_labels)
    #test_dataset = CustomDataset(test_src_tokens, None, 'test')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.batch_size, num_workers=1, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=hparams.batch_size, num_workers=1, shuffle=False)
    #test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=hparams.batch_size, num_workers=1, shuffle=False)

    # model
    if args.task == 'abs':
        model = AbsSummarizer(args, hparams.device, checkpoint=checkpoint_, bert_from_extractive=None)
    else:
        if args.only_transformer:
            model = ExtTransformer(args, hparams.device, checkpoint=None)
        else:
            model = ExtSummarizer(args, hparams.device, checkpoint=checkpoint_)
    print(model)


    def train_step(batch_item, epoch, batch, training):
        src = batch_item['src_token'].to(device)
        tar = batch_item['tar_token'].to(device)
        segs = batch_item['src_seg'].to(device)
        clss = batch_item['clss'].to(device)
        mask_src = batch_item['src_mask'].to(device)
        mask_tgt = batch_item['tar_mask'].to(device)
        mask_cls = batch_item['mask_cls'].to(device)
        labels = batch_item['src_sent_labels'].to(device)

        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        if training is True:
            model.train()
            optimizer.zero_grad()
            #model.zero_grad()
            with torch.cuda.amp.autocast():
                output, mask_ = model(src, tar, segs, clss, mask_src, mask_tgt, mask_cls)
                # output, _, _ = transformer([src, tar_inp, None])

                if args.task == 'abs':
                    loss = loss_function(tar_real, output)
            if args.task == 'ext':
                #print(labels)
                loss = BiCrossEntropy(output, labels.float())
                loss = (loss * mask_.float()).sum()
                acc = 0
                (loss / loss.numel()).backward()
            else:
                acc = accuracy_function(tar_real, output)
                loss.backward()
            optimizer.step()
            lr = optimizer.param_groups[0]["lr"]
            return loss, acc, round(lr, 10)
        else:
            model.eval()
            with torch.no_grad():
                output, mask_ = model(src, tar, segs, clss, mask_src, mask_tgt, mask_cls)
                # output, _, _ = transformer([src, tar_inp, None])

                if args.task == 'abs':
                    loss = loss_function(tar_real, output)
                else:
                    #print(labels)
                    loss = BiCrossEntropy(output, labels.float())
                    loss = (loss * mask_.float()).sum()
            if args.task == 'ext':
                acc = 0
            else:
                acc = accuracy_function(tar_real, output)
            return loss, acc, 0


    # opt, loss func
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)#hparams.learning_rate)
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
            #break # test
            total_loss += batch_loss
            total_acc += batch_acc
            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'LR' : lr,
                'Loss': '{:06f}'.format(batch_loss.item()),
                'Total Loss' : '{:06f}'.format(total_loss/(batch+1)),
                'Total ACC' : '{:06f}'.format(total_acc/(batch+1))
            })#;break # break test
            #if batch == 2:
            #    break
        #break # break test
        loss_plot.append(total_loss/(batch+1))
        acc_plot.append(total_acc/(batch+1))
        
        tqdm_dataset = tqdm(enumerate(val_dataloader))
        training = False
        for batch, batch_item in tqdm_dataset:
            #break # test
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
    parser.add_argument("-task", default='ext', type=str, choices=['ext', 'abs'])
    parser.add_argument("-only_transformer", type=bool, default=False)
    args = parser.parse_args()

    args.visible_gpus = '-1'
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.save_checkpoint_steps = 5

    args.temp_dir = './temp_dir'
    args.finetune_bert = True
    args.max_pos = 4000 # 512 #hparams.encoder_len # hparams.d_model #512
    args.share_emb = False
    args.use_bert_emb = False #True or False # False 해놓은 이유= baseline의 tgt vocab과 src vocab수가 다름.. 왜??
    args.sep_optim = True # seperate optimizer (encoder, decoder)
    args.accum_count = 1

    hidden_size = 512 # 768

    # encoder
    args.enc_dropout = 0.2
    args.enc_layers = 3 #6
    args.enc_hidden_size = hidden_size #hparams.encoder_len
    args.enc_ff_size = hparams.dff #dff or 512
    # decoder
    args.dec_dropout = 0.2 #0.2
    args.dec_layers = 3 #6
    args.dec_hidden_size = hidden_size #512 or 768
    args.dec_heads = 8
    args.dec_ff_size = hparams.dff #2048

    # params for extract
    args.ext_dropout = 0.2
    args.ext_layers = 2
    args.ext_hidden_size = hidden_size #512 or 768
    args.ext_heads = 8
    args.ext_ff_size = hparams.dff

    # weight init
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

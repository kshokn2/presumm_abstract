import numpy as np
import os
import argparse
from tqdm import tqdm

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


hparams = params.HParams()
device = hparams.device


def test(args, checkpoint):
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
    test_src_segs = src_tokenizer.token2seg(test_src_tokens)

    args.input_vocab_size = len(src_tokenizer.txt2idx)
    args.target_vocab_size = len(tar_tokenizer.txt2idx)

    train_dataset = CustomDataset(train_src_tokens, train_tar_tokens, train_src_segs)
    val_dataset = CustomDataset(val_src_tokens, val_tar_tokens, val_src_segs)
    test_dataset = CustomDataset(test_src_tokens, None, test_src_segs, 'test')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.batch_size, num_workers=1, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=hparams.batch_size, num_workers=1, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=hparams.batch_size, num_workers=1, shuffle=False)
    print('\n\n')

    model = AbsSummarizer(args, hparams.device, checkpoint=checkpoint, bert_from_extractive=None)
    print('load model..')

    def train_step(batch_item, epoch, batch, training):
        src = batch_item['src_token'].to(device)
        tar = batch_item['tar_token'].to(device)
        segs = batch_item['src_seg'].to(device)
        mask_src = batch_item['src_mask'].to(device)
        mask_tgt = batch_item['tar_mask'].to(device)

        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        if not True:
            model.train()
            #optimizer.zero_grad()
            model.zero_grad()
            with torch.cuda.amp.autocast():
                output, _ = model(src, tar, segs, None, mask_src, mask_tgt, None)
                # output, _, _ = transformer([src, tar_inp, None])
                #loss = loss_function(tar_real, output)
            acc = accuracy_function(tar_real, output)
            #loss.backward()
            #optimizer.step()
            lr = optimizer.param_groups[0]["lr"]
            return 0, acc, round(lr, 10)
        else:
            model.eval()
            with torch.no_grad():
                output, _ = model(src, tar, segs, None, mask_src, mask_tgt, None)
                # output, _, _ = transformer([src, tar_inp, None])
                #loss = loss_function(tar_real, output)
            acc = accuracy_function(tar_real, output)
            return 0, acc

    def accuracy_function(real, pred):
        accuracies = torch.eq(real, torch.argmax(pred, dim=2))
        mask = torch.logical_not(torch.eq(real, 0))
        accuracies = torch.logical_and(mask, accuracies)
        accuracies = torch.tensor(accuracies, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return torch.sum(accuracies)/torch.sum(mask)

    val_acc_plot = []

    if False: # True:
        total_val_loss, total_val_acc = 0, 0

        tqdm_dataset = tqdm(enumerate(val_dataloader))
        training = False
        for batch, batch_item in tqdm_dataset:
            batch_loss, batch_acc = train_step(batch_item, 0, batch, False)
            total_val_loss += batch_loss
            total_val_acc += batch_acc

            tqdm_dataset.set_postfix({
                'Epoch': 1,
                #'Val Loss': '{:06f}'.format(batch_loss.item()),
                'Total Val Loss' : '{:06f}'.format(total_val_loss/(batch+1)),
                'Total Val ACC' : '{:06f}'.format(total_val_acc/(batch+1))
            })
            break
        val_acc_plot.append(total_val_acc/(batch+1))


    def evaluate(batch_item):
        model.to(device)
        tokens = batch_item['src_token'].to(device)
        segs = batch_item['src_seg'].to(device)
        mask_src = batch_item['src_mask'].to(device)

        b_size = tokens.size(0)
        mask_tgt = None

        testing = 2
        if testing == 1:
            decoder_input = torch.full([b_size, hparams.decoder_len],
                                        tar_tokenizer.txt2idx['sos_'],
                                        dtype=torch.long,
                                        device=device)
            output = decoder_input;print(output.shape)
        elif testing == 0:
            #decoder_input = torch.tensor([tar_tokenizer.txt2idx['sos_']] * tokens.size(0), dtype=torch.long).to(device)
            #output = decoder_input.unsqueeze(1).to(device);print(output.shape)
            beam_size = 5
            alive_seq = torch.full([b_size * beam_size, 1],
                                        tar_tokenizer.txt2idx['sos_'],
                                        dtype=torch.long,
                                        device=device)
            decoder_input = alive_seq[:, -1].view(1, -1)
            output = decoder_input.transpose(0,1)

        '''
        for i in range(hparams.decoder_len-1):
            # predictions.shape == (batch_size, seq_len, vocab_size)
            with torch.no_grad():
                predictions, _ = model(tokens, output, segs, None, mask_src, mask_tgt, None);print(predictions,'\n')

            break
            # select the last token from the seq_len dimension
            #predictions_ = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

            #predicted_id = torch.tensor(torch.argmax(predictions_, axis=-1), dtype=torch.int32)

            #output = torch.cat([output, predicted_id], dim=-1)
        #output = output.cpu().numpy()

        summary_list = []
        token_list = []
        for token in output:
            summary = tar_tokenizer.convert(token)
            summary_list.append(summary)
            token_list.append(token)
        return summary_list, token_list
        '''

        with torch.no_grad():
            src_features = model.bert(tokens, segs, mask_src)
            dec_states = model.decoder.init_decoder_state(tokens, src_features, with_cache=True)

            # Tile states and memory beam_size times.
            beam_size = 1 #5
            alive_seq = torch.full([b_size * beam_size, 1],
                                        tar_tokenizer.txt2idx['sos_'],
                                        dtype=torch.long,
                                        device=device)
            #print(alive_seq.shape,alive_seq.size(1),alive_seq.size(0))
            #print(alive_seq[:, -1].view(1, -1))

            # Give full probability to the first beam on the first step.
            topk_log_probs = (
                torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                             device=device).repeat(b_size))


            for i in range(hparams.decoder_len-1):
                # Hmm..
                decoder_input = alive_seq[:, -1].view(1, -1)
                output = decoder_input.transpose(0,1)
                #print(output.shape, src_features.shape)

                # test..
                #output = alive_seq[:, -1]

                dec_out, dec_states = model.decoder(output, src_features, dec_states,
                                                     step=i)
                log_probs = model.generator.forward(dec_out.transpose(0,1).squeeze(0))
                #print(log_probs.shape)

                if beam_size > 1:
                    log_probs += topk_log_probs.view(-1).unsqueeze(1)

                alive_seq = torch.cat([alive_seq, torch.argmax(log_probs, dim=1).view(-1, 1)], dim=1)
                #if i == 1:
                #    break

            #print(alive_seq.shape)
            output_token = alive_seq.cpu().numpy()
            summary_list = []
            for token in output_token:
                summary = tar_tokenizer.convert(token)
                summary_list.append(summary)

        return alive_seq, summary_list

    def test_acc(real, pred):
        accuracies = torch.eq(real[:, 1:], pred[:, 1:])
        mask = torch.logical_not(torch.eq(real[:, 1:], 0))
        accuracies = torch.logical_and(mask, accuracies)
        accuracies = torch.logical_and(mask, accuracies)
        accuracies = torch.tensor(accuracies, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return torch.sum(accuracies)/torch.sum(mask)


    print('token_index)')
    print('pad_:',tar_tokenizer.txt2idx['pad_'], 'unk_:',tar_tokenizer.txt2idx['unk_'],'sos_:',tar_tokenizer.txt2idx['sos_'], 'eos_:',tar_tokenizer.txt2idx['eos_'], 'tgtsep_:',tar_tokenizer.txt2idx['tgtsep_'])
    print(f'{args.data}_dataloader')

    if args.data == 'train':
        train_dataset = CustomDataset(train_src_tokens, train_tar_tokens, train_src_segs)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.batch_size, num_workers=1, shuffle=True)
        tqdm_dataset = tqdm(enumerate(train_dataloader))
    elif args.data == 'val':
        val_dataset = CustomDataset(val_src_tokens, val_tar_tokens, val_src_segs)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=hparams.batch_size, num_workers=1, shuffle=False)
        tqdm_dataset = tqdm(enumerate(val_dataloader))
    else:
        test_dataset = CustomDataset(test_src_tokens, None, test_src_segs, 'test')
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=hparams.batch_size, num_workers=1, shuffle=False)
        tqdm_dataset = tqdm(enumerate(test_dataloader))

    preds = []
    acc1_preds = []
    tokens = []
    total_acc = 0
    for batch, batch_item in tqdm_dataset:
        #break
        output, summary = evaluate(batch_item)
        #print(batch_item['tar_token'][:,-1])
        tokens += output.cpu()
        preds += summary

        if args.data != 'test':
            accc = test_acc(batch_item['tar_token'].to(device), output)
            tqdm_dataset.set_postfix({
                'acc' : '{:06f}'.format(accc)
            })
            total_acc += accc
            if accc == 1.0 and acc1_preds == []:
                acc1_preds += summary
                '''
                print(output.shape);print(len(summary))
                sum_test = []
                for token in batch_item['tar_token'].cpu().numpy():
                    test = tar_tokenizer.convert(token)
                    sum_test.append(test)
                print(len(acc1_preds), len(sum_test), len(batch_item['tar_token']))
                if len(acc1_preds) ==len(sum_test):
                    for i in range(len(acc1_preds)):
                        print(sum_test[i])
                        print(acc1_preds[i])
                        print('==================================')
                        if i == 5:
                            break
                break
                '''
        #break
    if args.data != 'test':
        print(total_acc/(batch+1))
    print('done\n')

    if args.data != 'test':
        for i, (c, s, p) in enumerate(zip(df_val.context, df_val.summary, preds)):
            #break
            print('내용 :', c)
            print('=================================='.replace('=','-'))
            print('정답 :', s)
            print('예측 :', p)
            print('===================================================================')
            if i == 10:
                break
    else:
        for i, (c, p) in enumerate(zip(test.context, preds)):
            print('내용 :', c)
            print('=================================='.replace('=','-'))
            print('예측 :', p)
            print('===================================================================')
            if i == 10:
                break

    if args.submit and args.data == 'test':
        import pandas as pd
        submission = pd.read_csv('data/sample_submission.csv')
        submission['summary'] = preds
        submission.to_csv('data/dacon_baseline.csv', index=False)

        #from dacon_submit_api import dacon_submit_api 
        #result = dacon_submit_api.post_submission_file(
        #   '', # 파일경로
        #   '', # 개인토큰
        #   '', # 대회번호
        #   '', # 팀이름
        #   '') # 메모


if __name__ == '__main__':
    seed_everything(42)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-checkpoint_path", type=str)
    parser.add_argument("-data", default="test", type=str, choices=["train","val","test"])
    parser.add_argument("-submit", type=bool, default=False)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        setattr(args, k, opt[k])

    test(args, checkpoint)
    print('done')


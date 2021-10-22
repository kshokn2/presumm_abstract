import numpy as np
from tqdm import tqdm
from konlpy.tag import Mecab
import params

hparams = params.HParams()


class Mecab_Tokenizer():
    def __init__(self, max_length, mode, max_vocab_size=-1):
        self.text_tokenizer = Mecab()
        self.mode = mode
        self.txt2idx = {'pad_':0, 'unk_':1}#, 'sos_':2, 'eos_':3, 'cls_':4, 'sep_':5}
        self.idx2txt = {0:'pad_', 1:'unk_'}#, 2:'sos_', 3:'eos_', 4:'cls_', 5:'sep_'}
        self.max_length = max_length
        self.word_count = {}
        self.max_vocab_size = hparams.max_vocab_size
        self.encoder_len = hparams.encoder_len
        self.sep_vid = -1 # init val
        self.cls_vid = -1 # init val

        # 띄어쓰기를 찾기 위한 태그 목록
        self.font_blank_tag = [
            '', 'EC', 'EC+JKO', 'EF', 'EP+EC', 'EP+EP+EC', 'EP+ETM', 'EP+ETN+JKO', 'ETM', 'ETN', 'ETN+JKO', 'ETN+JX', 'IC', 'JC', 'JKB', 'JKB+JX', 'JKO',
            'JKQ', 'JKS', 'JX', 'MAG', 'MAG+JX', 'MAG+XSV+EP+EC', 'MAJ','MM', 'MM+EC', 'NNB', 'NNB+JKB', 'NNB+JKO', 'NNB+VCP+EC', 'NNBC', 'NNG', 'NNG+JX+JKO',
            'NNG+VCP+EC', 'NNP', 'NNP+JX', 'NP', 'NP+JKO', 'NP+JKS', 'NP+JX', 'NP+VCP+EC', 'NR', 'SC', 'SF', 'SL', 'SN', 'SSC', 'SSO', 'SY', 'UNKNOWN',
            'VA+EC', 'VA+EC+VX+ETM', 'VA+ETM', 'VA+ETN+JKB+JX', 'VCN+EC', 'VCN+ETM', 'VCP', 'VCP+EC', 'VCP+EP+EC', 'VCP+EP+ETM', 'VCP+ETM', 'VCP+ETN',
            'VV+EC', 'VV+EC+JX', 'VV+EC+VX+EC', 'VV+EC+VX+ETM', 'VV+EP+EC', 'VV+EP+ETM', 'VV+ETM', 'VV+ETN', 'VX+EC', 'VX+EC+VX+EP+EC', 'VX+EP+ETM',
            'VX+ETM', 'XPN', 'XR', 'XSA+EC', 'XSA+EC+VX+ETM', 'XSA+ETM', 'XSN', 'XSV+EC', 'XSV+EP+EC', 'XSV+ETM', 'XSV+ETN', 'XSV+JKO'
        ]
        self.back_blank_tag = [
            '', 'IC', 'MAG', 'MAG+JX', 'MAG+XSV+EP+EC', 'MAJ', 'MM', 'MM+EC', 'NNB', 'NNB+JKB', 'NNB+VCP', 'NNB+VCP+EC', 'NNB+VCP+EF', 'NNBC', 'NNBC+VCP+EC',
            'NNG', 'NNG+JC', 'NNG+JX+JKO', 'NNG+VCP', 'NNG+VCP+EC', 'NNG+VCP+ETM', 'NNP', 'NNP+JX', 'NP', 'NP+JKG', 'NP+JKO', 'NP+JKS', 'NP+JX', 'NP+VCP+EC', 'NP+VCP+EF',
            'NR', 'SC', 'SL', 'SN', 'SSC', 'SSO', 'SY', 'VA', 'VA+EC', 'VA+EC+VX+ETM', 'VA+EF', 'VA+ETM', 'VA+ETN', 'VA+ETN+JKB+JX', 'VCN', 'VCN+EC', 'VCN+EF', 'VCN+ETM',
            'VCN+ETN', 'VCP', 'VCP+EF', 'VV', 'VV+EC', 'VV+EC+JX', 'VV+EC+VX', 'VV+EC+VX+EC', 'VV+EC+VX+EF', 'VV+EC+VX+EP+EC', 'VV+EC+VX+ETM', 'VV+EF', 'VV+EP', 'VV+EP+EC',
            'VV+EP+ETM', 'VV+ETM', 'VV+ETN', 'VV+ETN+VCP+EF', 'VX', 'VX+ETM', 'XPN', 'XR', 'XSA+ETN+VCP+EF', 'XSN'
        ]

    def morpheme(self, sentence_list):
        new_sentence = []
        for i, sentence in tqdm(enumerate(sentence_list)):
            temp = []
            if self.mode == 'dec':
                temp.append('sos_')
            else:
                temp.append('cls_')
            for t in self.text_tokenizer.pos(sentence):
                temp.append('_'.join(t))
                if 'SF' in t:
                    if self.mode == 'dec':
                        # print(sentence)
                        temp.append('tgtsep_')
                    else:
                        temp.append('sep_')
                        temp.append('cls_')
            if self.mode == 'dec':
                if temp[-1] == 'tgtsep_':
                    temp = temp[:-1]
                temp.append('eos_')
            else:
                if temp[-1] == 'cls_':
                    temp = temp[:-1]
                else:
                    temp.append('sep_')
            new_sentence.append(' '.join(temp))

        return new_sentence

    def fit(self, sentence_list):
        for sentence in tqdm(sentence_list):
            for word in sentence.split(' '):
                # tokenize test
                #if self.mode == 'dec' and (word == 'cls_' or word == 'sep_'):
                #    continue
                #elif self.mode == 'enc' and (word == 'sos_' or word == 'eos_' or word == 'tgtsep_'):
                #    continue

                try:
                    self.word_count[word] += 1
                except:
                    self.word_count[word] = 1
        self.word_count = dict(sorted(self.word_count.items(), key=self.sort_target, reverse=True))

        self.txt2idx = {'pad_':0, 'unk_':1}#, 'sos_':2, 'eos_':3, 'cls_':4, 'sep_':5}
        self.idx2txt = {0:'pad_', 1:'unk_'}#, 2:'sos_', 3:'eos_', 4:'cls_', 5:'sep_'}
        if self.max_vocab_size == -1:
            for i, word in enumerate(list(self.word_count.keys())):
                self.txt2idx[word]=i+2 # i+2
                self.idx2txt[i+2]=word # tgt_words(??) # i+2
                if word == 'cls_' and self.cls_vid == -1:
                    self.cls_vid = self.txt2idx[word]
                elif word == 'sep_' and self.sep_vid == -1:
                    self.sep_vid = self.txt2idx[word]
        else:
            for i, word in enumerate(list(self.word_count.keys())[:self.max_vocab_size]):
                self.txt2idx[word]=i+2 # i+2
                self.idx2txt[i+2]=word # i+2
                if word == 'cls_' and self.cls_vid == -1:
                    self.cls_vid = self.txt2idx[word]
                elif word == 'sep_' and self.sep_vid == -1:
                    self.sep_vid = self.txt2idx[word]

    def sort_target(self, x):
        return x[1]

    def txt2token(self, sentence_list):
        tokens = []
        for sentence in tqdm(sentence_list):
            token = [0]*self.max_length
            for i, w in enumerate(sentence.split(' ')):
                if i == self.max_length:
                    break
                try:
                    token[i] = self.txt2idx[w]
                except:
                    token[i] = self.txt2idx['unk_']
            tokens.append(token)
        return np.array(tokens)

    # token2mask(for msk_src)
    def token2mask(self, token_list, types):
        if types == 'txt':
            mask = 1 - (token_list == 0)
        else:
            mask = 1 - (token_list == -1)
        return mask

    def token2seg(self, token_list):
        tok_seg = []
        for tokens in token_list:
            _segs = [-1] + [i for i, t in enumerate(tokens) if t == self.sep_vid]
            segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
            segments_ids = []
            for i, s in enumerate(segs):
                if (i % 2 == 0):
                    segments_ids += s * [0]
                else:
                    segments_ids += s * [1]
            if len(segments_ids) < self.encoder_len:
                segments_ids += [0] * (self.encoder_len - len(segments_ids))
            else:
                segments_ids = segments_ids[:self.encoder_len]
            tok_seg.append(segments_ids)

        return np.array(tok_seg)

    def convert(self, token):
        sentence = []
        for j, i in enumerate(token):
            if self.mode == 'enc':
                if all(i != self.txt2idx[tk] for tk in ['pad_', 'unk_', 'cls_', 'sep_']):
                    sentence.append(self.idx2txt[i].split('_')[0])
            elif self.mode == 'dec':
                if i == self.txt2idx['eos_'] or i == self.txt2idx['pad_']:
                    break
                elif i != 0:
                    if self.idx2txt[i] != 'tgtsep_':
                        sentence.append(self.idx2txt[i].split('_')[0])
                    else:
                        continue
                    # 앞뒤 태그를 확인하여 띄어쓰기 추가
                    if self.idx2txt[i].split('_')[1] in self.font_blank_tag:
                        try:
                            if self.idx2txt[token[j+1]].split('_')[1] in self.back_blank_tag:
                                sentence.append(' ')
                        except:
                            pass
        sentence = "".join(sentence)
        if self.mode == 'enc':
            sentence = sentence[:-1]
        elif self.mode == 'dec':
            sentence = sentence[3:-1]

        return sentence

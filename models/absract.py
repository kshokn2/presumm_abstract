from transformers import BertModel, BertConfig
import copy

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from models.decoder import TransformerDecoder, get_generator
import params


hparams = params.HParams()


class Bert(nn.Module):
    def __init__(self, temp_dir, finetune=False):
        super(Bert, self).__init__()
        # if(large):
        #     self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        # else:
        self.model = BertModel.from_pretrained("monologg/kobert", cache_dir=temp_dir)
        # self.model = BertModel.from_pretrained("bert-large-uncased", cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if(self.finetune):
            # if not put 'return_dict' in arg (return_dict=False), type of result is string
            # https://github.com/huggingface/transformers/pull/8530
            top_vec, _ = self.model(x, token_type_ids=segs, attention_mask=mask, return_dict=False)
            #print(last_hiddens, last_pooling_hiddens, hiddens)
            #top_vec = hiddens[-1]
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, token_type_ids=segs, attention_mask=mask, return_dict=False)
        return top_vec


class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.temp_dir, args.finetune_bert)

        # pretrained 'monologg/kobert' vocab_size -> 8002, so change this to our params..
        self.bert.model.config.vocab_size = args.input_vocab_size #target_vocab_size

        # if bert_from_extractive is not None:
        #     self.bert.model.load_state_dict(
        #         dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)

        # if (args.encoder == 'baseline'):
        bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
                                    num_hidden_layers=args.enc_layers, num_attention_heads=8,
                                    intermediate_size=args.enc_ff_size,
                                    hidden_dropout_prob=args.enc_dropout,
                                    attention_probs_dropout_prob=args.enc_dropout)
        self.bert.model = BertModel(bert_config)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = args.target_vocab_size #self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if(args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings;print(tgt_embeddings)
                self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)

        # decoder_outputs : batch*target_seq_len*dec_ff_size -> batch*target_seq_len*n_classes
        # self.generator = nn.Linear + sfmax (=last layer in Transfomer)
        decoder_outputs = self.generator(decoder_outputs)
        return decoder_outputs, None


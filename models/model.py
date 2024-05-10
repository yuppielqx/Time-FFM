from math import sqrt
import torch
import torch.nn as nn

from models.unitimegpt2 import UniTimeGPT2
from transformers import GPT2Tokenizer
import transformers
transformers.logging.set_verbosity_error()
class FlattenHead(nn.Module):
    def __init__(self, fea_num, pred_len, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(fea_num, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)#bs, channels, num_tokens*D
        x = self.linear(x)#bs, channels, pred_len
        x = self.dropout(x)
        return x

class ServerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mask_rate = args.mask_rate
        self.patch_len = args.patch_len
        self.max_backcast_len = args.max_backcast_len
        self.max_forecast_len = args.max_forecast_len
        self.logger = args.logger

        self.tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)#
        self.backbone = UniTimeGPT2.from_pretrained(args.model_path)#
        self.backbone.transformer.h = self.backbone.transformer.h[:args.lm_layer_num]#

        if args.lm_ft_type != 'full':#
            if args.lm_ft_type == 'freeze':
                words = []
            elif args.lm_ft_type == 'fpt':
                words = ['ln', 'wpe']
            else:
                exit(0)
            for name, param in self.backbone.named_parameters():
                flag = 0
                for w in words:
                    if w in name: flag = 1
                if flag:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                # print(name, param.shape, param.requires_grad)

        config = self.backbone.config#
        self.d_model = config.n_embd
        self.nhead = config.n_head
        self.dropout = config.attn_pdrop
        self.layer_norm_eps = config.layer_norm_epsilon

    def forward(self, info, embeds):
        x_enc = self.backbone(inputs_embeds=embeds)#

        return x_enc

class ClientEncoder(nn.Module):
    def __init__(self, args, word_embeddings):
        super().__init__()
        self.dynamic_prompt = args.dynamic_prompt
        self.mask_rate = args.mask_rate
        self.patch_len = args.patch_len
        self.max_backcast_len = args.max_backcast_len
        self.max_forecast_len = args.max_forecast_len
        self.logger = args.logger
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        #
        self.feature_embedding = nn.Linear(args.patch_len, self.d_model)#
        if args.mask_rate > 0:#
            self.feature_projection = nn.Linear(self.d_model, self.d_model)
            self.binary_indicator_embedding = nn.Linear(args.patch_len, self.d_model)
            self.gate_w1 = nn.Linear(self.d_model, self.d_model)
            self.gate_w2 = nn.Linear(self.d_model, self.d_model)
            self.gate_sigmoid = nn.Sigmoid()

        self.ts_embed_dropout = nn.Dropout(args.ts_embed_dropout)
        if self.dynamic_prompt:
            self.word_embeddings = word_embeddings
            self.vocab_size = self.word_embeddings.shape[0]
            self.num_tokens = args.num_tokens

            self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)#linear probing
            self.prompt_embedding = ReprogrammingLayer(self.d_model, n_heads=self.n_heads, topk=args.topk)

    def generate_ts_token(self, x_inp, seq_len, stride, mask):#
        if seq_len <= self.patch_len:
            ts_pad_num = self.patch_len - seq_len
        else:
            if seq_len % stride == 0:
                ts_pad_num = 0
            else:
                ts_pad_num = (seq_len // stride) * stride + self.patch_len - seq_len

        ts_padding = nn.ReplicationPad1d((0, ts_pad_num))
        x_inp = ts_padding(x_inp)
        mask = ts_padding(mask)

        x_inp = x_inp.unfold(dimension=-1, size=self.patch_len, step=stride)#
        mask = mask.unfold(dimension=-1, size=self.patch_len, step=stride)

        b, f, p, h = x_inp.shape
        x_inp = x_inp.reshape(b * f, p, h) #
        x_embed = self.feature_embedding(x_inp)#

        if self.mask_rate > 0:
            mask = mask.reshape(b * f, p, h)# batch_size * channels, num_patch, patch_len
            mask_embed = self.binary_indicator_embedding(mask)

            gate = self.gate_sigmoid(self.gate_w1(x_embed) + self.gate_w2(mask_embed))
            x_embed = gate * x_embed + (1 - gate) * mask_embed
            x_embed = self.feature_projection(x_embed)

        return self.ts_embed_dropout(x_embed), f


    def forward(self, info, x_inp, mask):
        data_id, seq_len, stride = info
        #对数据归一化
        means = torch.sum(x_inp, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_inp -= means
        x_inp = x_inp.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_inp * x_inp, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_inp /= stdev #batch_size * seq_len * channels

        x_inp = x_inp.transpose(1, 2)# batch_size * channel * seq_len
        mask = mask.transpose(1, 2)# batch_size * channel * seq_len
        x_token, n_vars = self.generate_ts_token(x_inp, seq_len, stride, mask)# batch_size * channels, num_patch, D

        if self.dynamic_prompt:
            source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
            embed_out = self.prompt_embedding(x_token, source_embeddings, data_id)
            inputs_embeds = torch.cat((embed_out, x_token), dim=1)
        else:
            inputs_embeds = x_token

        return inputs_embeds, means, stdev, n_vars


class ClientHead(nn.Module):
    def __init__(self, args, token_num):
        super().__init__()
        self.mask_rate = args.mask_rate
        self.patch_len = args.patch_len
        # self.token_num = token_num - args.topk
        self.token_num = token_num
        self.max_backcast_len = args.max_backcast_len
        self.max_forecast_len = args.max_forecast_len
        self.logger = args.logger
        self.d_model = args.d_model
        self.args = args
        self.dec_head = FlattenHead(fea_num=self.d_model * self.token_num,
                                    pred_len=args.max_backcast_len + args.max_forecast_len,
                                    head_dropout=args.dec_head_dropout)

    def forward(self, x_enc, means, stdev, n_vars):
        bs, token_num, _ = x_enc.shape
        x_dec = torch.reshape(
            x_enc, (-1, n_vars, x_enc.shape[-2], x_enc.shape[-1]))# bs, channels, num_tokens, D
        x_dec = x_dec.permute(0, 1, 3, 2) #bs, channels, D, num_tokens
        x_out = self.dec_head(x_dec)#bs, channels, pred_len
        x_out = x_out.transpose(2, 1)#bs, pred_len, channels,

        x_out = x_out * (stdev.repeat(1, x_out.shape[1], 1))
        x_out = x_out + (means.repeat(1, x_out.shape[1], 1))
        return x_out

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, topk, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_model // n_heads

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.n_heads = n_heads
        self.topk = topk

        self.out_projection = nn.Linear(d_keys * n_heads, d_model)

        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, data_id):
        #target_embedding: batch_size, num_patch, d_model
        #source_embedding: num_token, d_model
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)#B,L,H,d_keys
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)#S,H,d_keys

        out = self.reprogramming(target_embedding, source_embedding, data_id)#B,topk,H,E

        out = out.reshape(B, self.topk, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, data_id):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)#B,H,L,S

        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        tmps = torch.sum(A, axis=2)
        _, idxs = torch.topk(tmps, self.topk)

        t_s = source_embedding.repeat(B, 1, 1, 1).permute(0, 2, 1, -1)

        token_embedding = t_s[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], idxs, :]
        token_embedding = self.dropout(token_embedding.permute(0, 2, 1, -1))

        return token_embedding
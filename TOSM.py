##TOSM: topic_vec加上前一轮的，theta也加上前一轮的
##TOSM1: topic_vec不加前一轮的
import os
import time
import math
import random
import torch.utils.data
from torch_scatter import scatter
import torch.utils.data
from torch_geometric.data import DataLoader, Data
import argparse
import torch.utils.data
import torch
import torch.nn as nn
from torch_geometric.nn import GatedGraphConv, GraphConv, GCNConv
from torch.autograd import Variable
import numpy as np
from utils import VocabEntry
from collections import Counter

import torch.nn.functional as F
from torch.nn.init import xavier_uniform
from optimizer import DenseSparseAdam

from math import floor
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd
from get_dataset import MyDataset, MyData
from metric import get_precision, get_ndcg

clip_grad = 20.0
decay_epoch = 3
lr_decay = 0.1
max_decay = 5


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=3435)

    # device
    parser.add_argument('--device', default='cpu')  # do not use GPU acceleration
    # model
    parser.add_argument('--embed', type=int, default=300)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--hidden_size2', type=int, default=64)

    # topic_model
    parser.add_argument('--enc_nh', type=int, default=128)
    parser.add_argument('--num_topic', type=int, default=20)
    parser.add_argument('--ni', type=int, default=300)  # topic_vec的embedding dim
    parser.add_argument('--hidden', type=int, default=100)

    parser.add_argument('--nw', type=int, default=300)
    parser.add_argument('--prior', type=float, default=1.0)
    parser.add_argument('--STOPWORD', action='store_true', default=True)
    parser.add_argument('--nwindow', type=int, default=3)

    parser.add_argument('--variance', type=float, default=0.995)  # default variance in prior normal in ProdLDA

    parser.add_argument('--dropout', type=float, default=0.5)

    # model controller
    parser.add_argument('--fixing', action='store_true', default=True)
    parser.add_argument('--init_mult', type=float, default=1.0)
    # train
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--train_topic', type=int, default=5)
    parser.add_argument('--train_mlc', type=int, default=10)

    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.90)
    #parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--wdecay', type=float, default=0)
    parser.add_argument("--top_k", type=int, nargs="+", default=[5, 10, 15, 20, 30],
                    help="Cutoff values for computing metrics")
    parser.add_argument('--alpha', type=int, default=1)

    # 解析命令行参数，生成 args 对象
    args = parser.parse_args()


    # 尽早设置 PYTHONHASHSEED 环境变量，确保 Python 内部 hash 相关操作可复现
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # 固定 Python、NumPy 和 Torch 的随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 根据设备参数判断是否使用 CUDA，并设置 CUDA 随机种子及 cudnn 的相关参数
    if 'cuda' in args.device.lower() and torch.cuda.is_available():
        args.cuda = True
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    else:
        args.cuda = False

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return args


def kld_loss(z_post_mean, z_post_logvar, z_prior_mean=None, z_prior_logvar=None):
    device = args.device
    if z_prior_logvar is None:
        z_prior_logvar = torch.zeros(z_post_logvar.size()).to(device)
    if z_prior_mean is None:
        z_prior_mean = torch.zeros(z_post_mean.size()).to(device)
    z_post_var = torch.exp(z_post_logvar)
    z_prior_var = torch.exp(z_prior_logvar)
    klds_z = 0.5 * torch.sum(
        z_prior_logvar - z_post_logvar + (
                (z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / (z_prior_var)) - 1, dim=-1)
    return klds_z


def adj_mask(x_batch, device):
    size = torch.max(x_batch)
    N = x_batch.size(0)
    # mask = torch.ones((N, N), device=device)
    # mask = torch.dropout(mask,p=0.5, train=True)
    mask = torch.zeros((N, N), device=device)
    for i in range(size + 1):
        idxs = torch.where(x_batch == i)[0]
        mask[idxs[0]:idxs[-1] + 1, idxs] = 1
    mask2 = 1 - mask
    diag = torch.diag(mask)
    a_diag = torch.diag_embed(diag)
    mask = mask - a_diag
    return mask, mask2


class GNNDir2encoder(nn.Module):

    def __init__(self, args, word_vec):
        super(GNNDir2encoder, self).__init__()
        self.args = args
        if word_vec is not None:
            self.word_vec = word_vec
            if args.fixing:  # fixing=Ture,vector固定不变
                self.word_vec.requires_grad = False
        else:
            # self.word_vec = nn.Parameter(torch.Tensor(args.vocab, args.nw))
            self.word_vec = torch.eye(args.vocab_size, dtype=torch.float, device=args.device)
        input_size = self.word_vec.size(1)
        # self.enc1_gnn = GatedGraphConv(args.nw, num_layers=2, bias=True)  # 1995 -> 100
        self.enc1_gnn1 = GraphConv(input_size, args.nw, bias=True)
        self.bn_gnn1 = nn.BatchNorm1d(args.nw)
        # self.enc1_gnn2 = GraphConv(args.nw,args.nw,  bias=True)
        # self.bn_gnn2 = nn.BatchNorm1d(args.nw)

        self.enc2_fc1 = nn.Linear(input_size + args.nw, args.enc_nh)
        self.enc2_fc2 = nn.Linear(input_size + args.nw, args.enc_nh)
        self.enc2_drop = nn.Dropout(0.2)

        self.mean_fc = nn.Linear(args.enc_nh, args.num_topic)  # 100  -> 50
        self.mean_bn = nn.BatchNorm1d(args.num_topic)  # bn for mean
        self.logvar_fc = nn.Linear(args.enc_nh, args.num_topic)  # 100  -> 50
        self.logvar_bn = nn.BatchNorm1d(args.num_topic)  # bn for logvar

        self.phi_fc = nn.Linear(args.nw + input_size + args.enc_nh, args.num_topic)
        self.phi_bn = nn.BatchNorm1d(args.num_topic)

        self.logvar_bn.weight.requires_grad = False
        nn.init.constant_(self.logvar_bn.weight, 1.0)
        self.mean_bn.weight.requires_grad = False
        nn.init.constant_(self.mean_bn.weight, 1.0)

        prior_mean = torch.Tensor(1, args.num_topic).fill_(0)
        prior_var = torch.Tensor(1, args.num_topic).fill_(args.variance)
        # self.a = args.prior * np.ones((1, args.num_topic)).astype(np.float32)
        # prior_mean = torch.from_numpy((np.log(self.a).T - np.mean(np.log(self.a), 1)).T)
        # prior_var = torch.from_numpy((((1.0 / self.a) * (1 - (2.0 / args.num_topic))).T +
        #                               (1.0 / (args.num_topic * args.num_topic) * np.sum(1.0 / self.a, 1)).T))
        # prior_mean = torch.Tensor(1, args.num_topic).fill_(0)
        #
        # prior_var = torch.Tensor(1, args.num_topic).fill_(1.0 / self.args.prior * (1 - 1.0/args.num_topic))
        prior_logvar = prior_var.log()
        self.register_buffer('prior_mean', prior_mean)
        self.register_buffer('prior_var', prior_var)
        self.register_buffer('prior_logvar', prior_logvar)

        # self.propa = GCNConv(args.num_topic, args.num_topic, bias=False)
        # nn.init.eye_(self.propa.weight)
        # self.propa.weight.requires_grad=False

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.logvar_fc.weight)
        nn.init.zeros_(self.logvar_fc.bias)
        pass

    def forward(self, idx_x, idx_w, x_batch, edge_index, edge_weight):
        x = self.word_vec[idx_x]  # N*nw
        diag = torch.ones(2, idx_x.size(0), dtype=torch.long, device=idx_x.device).cumsum(dim=-1) - 1
        edge_index_exp = torch.cat([edge_index, diag], dim=-1)  # 添加对角线
        diag_w = torch.ones(idx_x.size(0), dtype=torch.float, device=idx_x.device) * idx_w
        edge_weight_exp = torch.cat([edge_weight, diag_w], dim=0)
        enc1 = torch.tanh(self.bn_gnn1(self.enc1_gnn1(x, edge_index_exp, edge_weight=edge_weight_exp)))

        if torch.isnan(enc1).sum() > 0:
            import ipdb
            ipdb.set_trace()
        enc1 = torch.cat([enc1, x], dim=-1)
        enc2 = torch.sigmoid(self.enc2_fc1(enc1)) * torch.tanh(self.enc2_fc2(enc1))

        size = int(x_batch.max().item() + 1)
        enc2 = scatter(enc2, x_batch, dim=0, dim_size=size, reduce='sum')  # B*enc_h

        enc2d = self.enc2_drop(enc2)

        mean = self.mean_bn(self.mean_fc(enc2d))  # posterior mean
        logvar = self.logvar_fc(enc2d)  # posterior log variance

        word_embed = torch.cat([enc1, enc2[x_batch]], dim=-1)
        phi = torch.softmax(self.phi_fc(word_embed), dim=-1)  # (B*max_len)*num_topic

        param = (mean, logvar)
        if torch.isnan(mean).sum() > 0:
            import ipdb
            ipdb.set_trace()
        return param, phi

    def reparameterize(self, param):
        posterior_mean = param[0]
        posterior_var = param[1].exp()
        # take sample
        if self.training:
            eps = torch.zeros_like(posterior_var).normal_()
            z = posterior_mean + posterior_var.sqrt() * eps  # reparameterization
        else:
            z = posterior_mean
        theta = torch.softmax(z, dim=-1)
        return theta

    def KL_loss(self, param):
        posterior_mean = param[0]
        posterior_logvar = param[1]

        prior_mean = Variable(self.prior_mean).expand_as(posterior_mean)
        prior_var = Variable(self.prior_var).expand_as(posterior_mean)
        prior_logvar = Variable(self.prior_logvar).expand_as(posterior_mean)
        var_division = posterior_logvar.exp() / prior_var
        diff = posterior_mean - prior_mean
        diff_term = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        KL = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.args.num_topic)

        if torch.isinf(KL).sum() > 0 or torch.isnan(KL).sum() > 0:
            import ipdb
            ipdb.set_trace()

        return KL


class GSM(nn.Module):

    def __init__(self, args, word_vec=None, params=None):
        super(GSM, self).__init__()
        self.args = args
        self.params = params
        # graph_encoder
        self.graph_encoder = GNNDir2encoder(self.args, word_vec)

        # encoder
        self.enc1_fc = nn.Linear(args.vocab_size, 2 * args.enc_nh)
        self.enc2_fc = nn.Linear(2 * args.enc_nh, args.enc_nh)
        self.en2_drop = nn.Dropout(0.2)
        self.mean_fc = nn.Linear(args.enc_nh, args.num_topic)
        self.logvar_fc = nn.Linear(args.enc_nh, args.num_topic)

        # fasttext encoder
        self.words_dim = word_vec.shape[1]
        self.embed = nn.Embedding.from_pretrained(word_vec, freeze=False)
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(self.words_dim, args.enc_nh)

        # decoder
        self.decoder = nn.Linear(args.num_topic, args.num_topic)

        # topic parameter
        if word_vec is None:
            self.word_vec = nn.Parameter(torch.Tensor(args.vocab_size, args.ni))  # [1384,300]
            nn.init.normal_(self.word_vec, std=0.01)
        else:
            self.word_vec = nn.Parameter(word_vec.clone().detach().requires_grad_(True))  # word_vector更新
        self.topic_vec = nn.Parameter(torch.Tensor(args.num_topic, args.ni))

        # initialize decoder weight
        if args.init_mult != 0:
            # std = 1. / math.sqrt( ac.init_mult * (ac.num_topic + ac.num_input))
            self.decoder.weight.data.uniform_(0, args.init_mult)

        self.enc_params = list(self.graph_encoder.parameters()) + list(self.enc1_fc.parameters()) + list(
            self.enc2_fc.parameters()) + list(
            self.mean_fc.parameters()) \
                          + list(self.logvar_fc.parameters()) + list(self.decoder.parameters())
        self.dec_params = [self.word_vec, self.topic_vec]
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.logvar_fc.weight)
        nn.init.zeros_(self.logvar_fc.bias)
        nn.init.normal_(self.topic_vec)
        # nn.init.normal_(self.word_vec, std=0.01)

    def set_embedding(self, embedding, fix=False):
        assert embedding.size() == self.word_vec.size()
        self.word_vec = nn.Parameter(embedding)
        if fix:
            self.word_vec.requires_grad = False

    def forward(self, batch_data, input, y, epoch):  # input=text_bow
        # graph_encoder
        # compute posterior 由assessment计算后验概率
        param, phi = self.graph_encoder(batch_data.x, batch_data.x_w, batch_data.x_batch, batch_data.edge_index,
                                        batch_data.edge_w)  # phi表示每个词的词分布
        # phi (B*max_len)*num_topic
        self.posterior_mean = param[0]
        posterior_logvar = param[1]
        posterior_var = posterior_logvar.exp()

        # compute posterior
        enc1 = torch.tanh(self.enc1_fc(input))  # enc1_fc   output
        enc2 = torch.tanh(self.enc2_fc(enc1))  # encoder2 output

        self.prior_mean = self.mean_fc(enc2)  # posterior mean
        prior_logvar = self.logvar_fc(enc2)  # posterior log variance
        prior_var = prior_logvar.exp()

        # take sample
        ia_bow = to_BOW(batch_data.x, batch_data.x_w, batch_data.x_batch, self.args.vocab_size)
        eps = Variable(ia_bow.data.new().resize_as_(self.posterior_mean.data).normal_())  # noise
        if self.training:
            z = self.posterior_mean + posterior_var.sqrt() * eps  # reparameterization #主题分布
        else:
            z = self.posterior_mean
        p = z
        # do reconstruction
        beta = self.get_beta()
        if self.training:
            theta = torch.softmax(self.decoder(p), dim=-1)  # 文档对应的主题分布  #单词对应的主题分布 #theta:[batch,num_topic]
        else:
            theta = torch.softmax(self.decoder(self.prior_mean), dim=-1)

        if int(epoch) < args.train_topic:
            recon = torch.matmul(theta, beta)
            NL = -(input * (recon + 1e-10).log()).sum(1)
            KL1 = kld_loss(self.posterior_mean, posterior_logvar, self.prior_mean, prior_logvar)
            KL2 = -0.5 * torch.sum(1 - self.posterior_mean ** 2 + posterior_logvar -
                                   torch.exp(posterior_logvar), dim=1)

            recon_structure = torch.tensor(0, dtype=torch.float32)



        else:
            beta_s = beta[:, batch_data.x]  # K*(B*len)
            beta_s = beta_s.permute(1, 0)
            recon_word = (phi * (beta_s + 1e-6).log()).sum(-1)  # (B*len)
            NL = -scatter(batch_data.x_w * recon_word, index=batch_data.x_batch, dim=-1, dim_size=args.batch_size,
                          reduce='sum')  # B
            KL1 = kld_loss(self.posterior_mean, posterior_logvar, self.prior_mean, prior_logvar)
            a = torch.sum(phi * ((phi / (theta[batch_data.x_batch] + 1e-10) + 1e-10).log()), dim=-1)  # (B*len)
            KL2 = scatter(a, index=batch_data.x_batch, dim=-1, dim_size=args.batch_size, reduce='sum')  # B

            # graph_VGAE
            p_phi = phi[batch_data.edge_index, :]  # 2,B*len_edge, K
            p_edge = torch.sigmoid(torch.sum((p_phi[0] * p_phi[1]), dim=-1)).log()  # B*len_edge
            p_edge = scatter(p_edge, batch_data.edge_id_batch, dim=0, dim_size=args.batch_size, reduce='sum')  # B
            neg_mask, neg_mask2 = adj_mask(batch_data.x_batch, device=batch_data.x_w.device)
            neg_mask[batch_data.edge_index[0], batch_data.edge_index[1]] = 0
            n_edge = torch.sigmoid(torch.matmul(phi, phi.T)).log()  # B*len, B*len
            n_edge1 = torch.sum(n_edge * neg_mask, dim=-1)  # B*len
            n_edge1 = scatter(n_edge1, batch_data.x_batch, dim=0, dim_size=args.batch_size, reduce='sum')  # B
            tmp = torch.ones_like(batch_data.edge_id_batch, dtype=torch.float, device=batch_data.edge_id_batch.device)
            NP = scatter(tmp, batch_data.edge_id_batch, dim=0, dim_size=args.batch_size, reduce='sum')  # B
            NN = scatter(torch.sum(neg_mask, dim=-1), batch_data.x_batch, dim=0, dim_size=args.batch_size,
                         reduce='sum')  # B
            recon_structure = -(p_edge + n_edge1 / (NN + 1e-6) * NP)

            if args.multiround == '1':
                #theta=self.prior_mean
                theta = theta
            elif args.multiround == '2':
                #theta = torch.cat((theta, self.params[0]), 1)
                theta = theta
            else:
                #theta1 = torch.cat((theta, self.params[0]), 1)
                #theta = torch.cat((theta1, self.params[2]), 1)  # [batch*150]
                theta = theta


        # loss
        KL = KL1 + KL2
        topic_loss = (NL + KL + recon_structure).mean()

        outputs = {
            "loss": topic_loss.mean(),
            "recon_word": NL.mean(),
            "KL1": KL1.mean(),
            "KL2": KL2.mean(),
            "recon_structure": recon_structure.mean(),
        }
        return outputs, theta,topic_loss

    def get_beta(self):
        beta = torch.softmax(torch.matmul(self.topic_vec, self.word_vec.T), dim=-1)
        return beta

    def loss(self, input, y):
        return self.forward(input, y)


class Topic_Att(nn.Module):

    def __init__(self, args, word_vec, params):
        super().__init__()
        target_class = word_vec.shape[0]
        words_dim = word_vec.shape[1]
        self.args = args
        self.word_vec = word_vec[1:, :]
        self.mlb = args.mlb
        self.params=params

        # topic model
        self.vtm = GSM(self.args, word_vec=word_vec, params=self.params)

        # attention
        self.fc_att = nn.Linear(args.hidden_size * 2, args.ni)
        self.dropout = nn.Dropout(args.dropout)

        # LSTM
        self.embedding = nn.Embedding.from_pretrained(word_vec, freeze=False)

        self.lstm = nn.LSTM(args.embed, args.hidden_size, args.num_layers,
                            bidirectional=True, batch_first=True, dropout=args.dropout)
        self.tanh1 = nn.Tanh()
        # whole
        self.fc = nn.Linear(args.hidden_size * 2, args.num_labels)
        self.sigmoid = nn.Sigmoid()
        self.loss_func = torch.nn.BCELoss()

    def forward(self, batch_data, epoch, **kwargs):
        # jd+r
        text = batch_data.text
        text_batch = batch_data.text_batch
        text_w = batch_data.text_w
        x = get_batch_idx(text, text_batch)
        text_bow = to_BOW(text, text_w, text_batch, args.vocab_size)  # [batch,1384]

        # label
        idx_sent = batch_data.idx_sent
        idx_sent_batch = batch_data.idx_sent_batch
        y = get_batch_idx(idx_sent, idx_sent_batch)

        new_x = get_batch_idx(idx_sent, idx_sent_batch)
        target = new_x.cpu().numpy()
        target = self.mlb.transform(target)
        # target = target.to(torch.float32).to(args.device)
        target = torch.from_numpy(target).to(torch.float32).to(args.device)
        # print(target.shape)
        # print(target)

        # Bi-LSTM
        emb = self.embedding(x)  # (batch, sent_len, embed_dim)
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        # M = self.tanh1(H)  # [128, 32, 256]
        values = H  # [batch_size, seq_len, hidden_size * 2]

        # topic attention
        _, w, topic_loss = self.vtm.forward(batch_data, text_bow, y, epoch)  # w： [batch,num_topic]

        if args.multiround == '1':
            topic_vec = self.vtm.topic_vec
        elif args.multiround == '2':
            #topic_vec = torch.cat((self.vtm.topic_vec, self.params[1]), 0)
            topic_vec = self.vtm.topic_vec
        else:
            #topic_vec1 = torch.cat((self.vtm.topic_vec, self.params[1]), 0)
            #topic_vec = torch.cat((topic_vec1, self.params[3]), 0)  # [batch*150]
            topic_vec = self.vtm.topic_vec
            
        #topic_embed_unstack = torch.unbind(self.vtm.topic_vec)  # a list of topic vectors
        topic_embed_unstack = torch.unbind(topic_vec)

        topic_atten_weights = []
        self.h1 = torch.tanh(self.fc_att(values))  # [batch_size, seq_len, args.ni]
        # print('h1',self.h1.shape)#[32,122,300]
        for i in range(w.shape[-1]):
            query = topic_embed_unstack[i]  # [args.ni,]
            # print('query',query.shape)#[300]
            b = torch.mul(self.h1, query)  # [batch_size, seq_len,args.ni ]
            # print('b',b.shape)#[32,122,300]

            score = torch.sum(torch.mul(self.h1, query), dim=-1, keepdim=True)  # [batch_size, seq_len,1]
            # print('score',score.shape)#[32,122]
            attention_weights = torch.softmax(score, dim=1)  # [batch_size, seq_len,1]
            topic_atten_weights.append(attention_weights)

        topic_atten = torch.matmul(torch.cat(topic_atten_weights, -1), torch.unsqueeze(w, -1))
        # [batch,seq_len,1]=[batch_size, seq_len,num_topic]*[batch,num_topic,1]
        atten_out = torch.sum(torch.mul(topic_atten, values), dim=1)  # [batch,hidden_size * 2]
        
        #atten_out = F.relu(atten_out)
        atten_out_dropped = self.dropout(atten_out)
        out = self.fc(atten_out_dropped)
        logit = self.sigmoid(out)

        # loss
        cross_entro = self.loss_func(logit, target)

        loss = (args.alpha*cross_entro + topic_loss)
        return logit, loss


def get_max_lenth(split_tensor):
    lenth = []
    for unit in split_tensor:
        lenth.append(len(unit))
    max_lenth = max(lenth)
    return max_lenth


def get_batch_idx(idx, idx_batch):
    idx_batch = idx_batch.cpu().numpy().tolist()
    d = Counter(idx_batch)
    d_s = sorted(d.items(), key=lambda x: x[0], reverse=False)
    batch_section = np.zeros(args.batch_size).astype(int)
    for i in d_s:
        batch_section[i[0]] = i[1]
    split_tensor = torch.split(idx, batch_section.tolist(), dim=0)
    masks = []
    for unit in split_tensor:
        mask = np.zeros(get_max_lenth(split_tensor))
        s_len = len(unit)
        mask[:s_len] = unit.cpu()
        masks.append(mask)
    return torch.tensor(masks, dtype=torch.long).to(args.device)


def get_word_vec(word_vector_file, vocab_path):
    vocab = VocabEntry.from_corpus(vocab_path, withpad=True)
    vector_dic = {}
    with open(word_vector_file, encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split()
            word = line[0]
            vector = [np.float32(val) for val in line[1:]]
            # print(len(vector))
            vector_dic[word] = vector
    file.close()
    vectors = []
    word_embed_dim = 300
    for i in range(0, len(vocab)):
        if i == 0:
            vectors.append(np.zeros(word_embed_dim))
        else:
            word = vocab.id2word(i)
            if word not in vector_dic.keys():
                vectors.append(np.random.uniform(-0.25, 0.25, word_embed_dim))
                # vectors.append(np.zeros(word_embed_dim))

            else:
                word_vector = vector_dic[word]
                # print(len(word_vector))
                vectors.append(word_vector)
    embedding = torch.FloatTensor(vectors)
    return embedding


def test(model, test_loader, epoch, mode='VAL'):
    val_loss = 0
    # 针对每个 top_k 值，使用字典存储各个指标的列表
    val_pre = {k: [] for k in args.top_k}
    val_recall = {k: [] for k in args.top_k}
    val_f1 = {k: [] for k in args.top_k}
    val_ndcg = {k: [] for k in args.top_k}

    model.eval()
    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(args.device)
            # interview 的文本
            idx_sent = batch_data.idx_sent
            idx_sent_batch = batch_data.idx_sent_batch
            
            new_x = get_batch_idx(idx_sent, idx_sent_batch)         # ground truth 标签索引

            logits, loss = model(batch_data, epoch)  # logits: [batch, labels_num]
            val_loss += loss.item()

            # 针对每个 top_k 值分别计算指标
            for top_k in args.top_k:
                # 获取预测分数最高的 top_k 标签（torch.topk 返回 scores 和 indices）
                scores, tmp = torch.topk(logits, top_k)
                # tmp: [batch, top_k]，表示预测出的标签位置索引
                batch_pred = tmp.cpu().numpy()
                batch_targets = new_x.cpu().numpy()  # ground truth

                # 计算指标，假设 get_precision 和 get_ndcg 接受单个 top_k 值
                pr, rec, f1 = get_precision(args.mlb, batch_pred, batch_targets, top_k)
                ndcg = get_ndcg(args.mlb, batch_pred, batch_targets, top_k)

                val_pre[top_k].append(pr)
                val_recall[top_k].append(rec)
                val_f1[top_k].append(f1)
                val_ndcg[top_k].append(ndcg)

    val_loss /= len(test_loader)

    # 对每个 top_k 计算平均指标，并打印输出
    final_metrics = {}
    for top_k in args.top_k:
        pres = np.array(val_pre[top_k]).mean()
        recs = np.array(val_recall[top_k]).mean()
        f1s = np.array(val_f1[top_k]).mean()
        ndcgs = np.array(val_ndcg[top_k]).mean()
        final_metrics[top_k] = {
            "precision": pres,
            "recall": recs,
            "f1": f1s,
            "ndcg": ndcgs
        }
        print('{} --- top_k={} --- loss:{:.4f}, ndcg:{:.4f}, f1:{:.4f}, precision:{:.4f}, recall:{:.4f}'
              .format(mode, top_k, val_loss, ndcgs, f1s, pres, recs))

    return val_loss, final_metrics



def get_topic(predict, topk, vocab_entity):
    _, index = torch.topk(predict, topk, 1)  # predict为tensor
    topics = []
    for i in index:
        words = []
        for j in i:
            word = vocab_entity.id2word(j)
            words.append(word)
        topics.append(words)
    return topics  # [outputsize,topk]


def mapone_hot(inst_map):
    new_mask_image = torch.zeros([inst_map.shape[0], inst_map.shape[1]], dtype=torch.float32)
    for i in range(inst_map.shape[0]):
        f_mask = inst_map[i, :]
        zero = torch.zeros_like(f_mask)
        one = torch.ones_like(f_mask)
        f_mask = torch.where(f_mask >= 1, one, zero)
        new_mask_image[i, :] = f_mask
    return new_mask_image


def to_BOW(idx, idx_w, idx_batch, V):
    device = idx.get_device()
    if device >= 0:
        embeddings = torch.eye(V, device=device)
    else:
        embeddings = torch.eye(V)
    batch_data = embeddings[idx]
    batch_data = torch.unsqueeze(idx_w, 1) * batch_data
    # print('idx_batch.max().item()',idx_batch.max().item())
    # print(idx_batch)
    # size = int(idx_batch.max().item() + 1)
    size = args.batch_size
    # print('size',size)
    batch_data = scatter(batch_data, index=idx_batch, dim=-2, dim_size=size, reduce='sum')
    return batch_data


def get_mlb(fname, vocab, round_num):
    exp_data = pd.read_csv(fname, sep='\t', encoding='utf-8')
    round_dic = {'1': 'assessment_skill1', '2': 'assessment_skill2', '3': 'assessment_skill3'}
    #round_dic = {'1': 'unique1', '2': 'unique2', '3': 'unique3'}
    round_data = round_dic[round_num]

    a = [sorted(set(str(assess).split(','))) for assess in exp_data[round_data]]
    labels = []
    for i in a:
        label_idxs = [vocab[word] for word in i if vocab[word] > 0]
        labels.append(label_idxs)
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)
    return mlb



def main(args):
    device = torch.device(args.device)

    vocab_path = '../TOSM/exp_data/jra_label.txt'
    # Glove_vectors
    word_vector_file = '../TOSM/exp_data/Glove_vectors_300.txt'
    # 数据集
    dataset_path = '../TOSM/exp_data/three_rounds.tsv'

    processed_data_path = '../TOSM/processed_data'
    save_root = '../TOSM/saved_model'
    params = []
    model_name="TOSM"

    for round_num in ['1', '2', '3']:
        args.multiround = round_num
        #begin = time.time()
        print('Model name:{}-seed:{}-num_epoch:{}-Start round:{}'.format(model_name,args.seed,args.num_epoch,args.multiround))
        save_path = os.path.join(save_root, 'Model name:{}-seed:{}-num_epoch:{}-round{}'.format(model_name,args.seed,args.num_epoch,args.multiround))

        dataset = MyDataset(processed_data_path, dataset_path, vocab_path, ngram=args.nwindow, STOPWORD=args.STOPWORD,
                            edge_threshold=3,round=round_num)
        train_idxs = [i for i in range(len(dataset)) if dataset[i].split == 0]
        train_data = dataset[train_idxs]
        val_idxs = [i for i in range(len(dataset)) if dataset[i].split == 1]
        val_data = dataset[val_idxs]
        test_idxs = [i for i in range(len(dataset)) if dataset[i].split == 2]
        test_data = dataset[test_idxs]
        vocab = dataset.vocab
        args.vocab = vocab
        args.vocab_size = len(vocab)  # 总词表数量
        args.mlb = get_mlb(dataset_path, args.vocab, round_num)
        args.num_labels = len(args.mlb.classes_)  # 预测标签的数量

        training_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False,
                                     follow_batch=['x', 'edge_id', 'text', 'text_sent', 'idx_sent'], drop_last=True)
        deving_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                                   follow_batch=['x', 'edge_id', 'text', 'text_sent', 'idx_sent'],
                                   drop_last=True)
        testing_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                    follow_batch=['x', 'edge_id', 'text', 'text_sent', 'idx_sent'], drop_last=True)

        word_vectors = get_word_vec(word_vector_file, vocab_path).to(args.device)
        print('vocab_size:', args.vocab_size)
        print('num_labels:', args.num_labels)

        model = Topic_Att(args, word_vec=word_vectors, params=params)
        model.to(device)

        #parameter = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.wdecay)
        #optimizer = DenseSparseAdam(params=model.parameters(), lr=args.learning_rate)
        iterations = 0
        log_niter = len(training_loader) // 10
        best_dev_f1 = 0
        decay_cnt = 0
        opt_dict = {"not_improved": 0, "lr": args.learning_rate, "best_dev_f1": 0}
        best_dev_f1 = 0
        best_epoch = 0
        iterations = 0

        for epoch in range(args.num_epoch):
            print(epoch, '--------------------------------------------------------')
            num_sents = 0
            total_loss = 0

            # 开始训练模型
            model.train()
            for batch_idx, batch_data in enumerate(training_loader):
                iterations += 1
                optimizer.zero_grad()

                batch_data = batch_data.to(args.device)
                scores, loss = model(batch_data, epoch)
                loss.backward()
                optimizer.step()
                total_loss += loss * args.batch_size
                num_sents += args.batch_size

                if iterations % log_niter == 0:
                    print('epoch:{},loss:{:.4f}'.format(epoch, total_loss / num_sents))

            # 开始验证模型
            val_loss, final_metrics = test(model, deving_loader, epoch, 'VAL')
            f1_at_30 = final_metrics[30]['f1']  # 计算F1@30

            # 进行早停判断
            if f1_at_30 > best_dev_f1:
                best_dev_f1 = f1_at_30
                best_epoch = epoch
                torch.save(model.state_dict(), save_path)  # 保存最优模型
                opt_dict["not_improved"] = 0  # 重置未改进计数
            else:
                opt_dict["not_improved"] += 1
                if opt_dict["not_improved"] >= decay_epoch:  # 如果达到衰减阈值
                    opt_dict["best_dev_f1"] = best_dev_f1  # 保持当前最优 f1
                    opt_dict["not_improved"] = 0
                    # 可选：可以在这里应用学习率衰减策略
                    # 学习率衰减：将学习率衰减为原来的 lr_decay 倍
                    #for param_group in optimizer.param_groups:
                    #    param_group['lr'] *= lr_decay
                    #print(f"Learning rate decayed to {optimizer.param_groups[0]['lr']:.6f}")
                    decay_cnt += 1

            # Early Stopping 判断
            if decay_cnt == max_decay:
                print("Early Stopping.")
                print('best_epoch:%d \t best_dev_f1:%.4f' % (best_epoch, opt_dict["best_dev_f1"]))
                break

        # 在训练结束后加载最优模型并在测试集上进行评估
        model.load_state_dict(torch.load(save_path))
        test_loss, final_metrics = test(model, testing_loader, 1000, 'TEST')  # 测试集评估

        # 打印最终的测试集表现
        print("Final Test Metrics:")
        for top_k in args.top_k:
            print(f"Top {top_k}: Precision = {final_metrics[top_k]['precision']:.4f}, "
                f"Recall = {final_metrics[top_k]['recall']:.4f}, F1 = {final_metrics[top_k]['f1']:.4f}, "
                f"NDCG = {final_metrics[top_k]['ndcg']:.4f}")
            

        params.append(model.vtm.posterior_mean) #加入前一轮的theta
        params.append(model.vtm.topic_vec)  ##加入前一轮的topic
        #torch.cuda.empty_cache()

    # 输出结果
    #print(params)



if __name__ == '__main__':
    args = init_config()
    main(args)


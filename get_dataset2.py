#不加上一轮interview的
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from collections import Counter
from utils import VocabEntry
import math
import os


def D_con(pair, pair_list):  # pair_list=pair_t tuple形式
    D_con = 0
    for i in pair_list:
        if pair in i:
            D_con += 1
    return D_con


def tf(pair, count_dic):  # count_dic=coun_dct
    return count_dic[pair] / sum(count_dic.values())


def idf(pair, pair_list):  # pair_list=pair_t
    return math.log(len(pair_list)) / (1 + D_con(pair, pair_list))


def tfidf(pair, count_dic, pair_list):
    return tf(pair, count_dic) * idf(pair, pair_list)


class PreTextData(object):
    """docstring for MonoTextData"""

    def __init__(self, fname, ngram=3, vocab=None, edge_threshold=10, round='1'):
        super(PreTextData, self).__init__()

        self.label_data, self.jd_resume_data, self.vocab, self.word_count, self.itemids, self.split= self._read_corpus(
            fname, vocab, round)
        self.ngram = ngram
        self.vocab = vocab
        self.pairVocab(edge_threshold)

    def __len__(self):
        return len(self.label_data)

    def _read_corpus(self, fname, vocab: VocabEntry, round='1'):
        label_data = []
        jd_resume_data = []
        itemids = []
        split = []
        label_index = []
        word_count = 0
        csvdata = pd.read_csv(fname, sep='\t', encoding='utf-8')
        round_dic = {'1': 'assessment_skill1', '2': 'assessment_skill2', '3': 'assessment_skill3'}
        round_data = round_dic[round]

        if round == '1':
            for i, ss in enumerate(
                    csvdata[['index', round_data, 'jd_skill', 'resume_skill', 'train']].values):
                label_line = str(ss[1])
                words = label_line.strip().split(',')
                idxs = [vocab[word] for word in words if vocab[word] > 0]
                word_num = len(idxs)
                label_data.append(idxs)
                itemids.append(ss[0])
                word_count += word_num

                jr = []

                jd_line = str(ss[2])
                jd_skill = jd_line.strip().split(',')
                jd_idxs = [vocab[word] for word in jd_skill if vocab[word] > 0]
                jr.extend(jd_idxs)

                resume_line = str(ss[3])
                resume_skill = resume_line.strip().split(',')
                resume_idxs = [vocab[word] for word in resume_skill if vocab[word] > 0]
                jr.extend(resume_idxs)

                jd_resume_data.append(jr)

                split.append(int(ss[4]))


        elif round == '2':
            for i, ss in enumerate(csvdata[['index', round_data, 'jd_skill', 'resume_skill', 'train', 
                                            'assessment_skill1']].values):
                label_line = str(ss[1])
                words = label_line.strip().split(',')
                idxs = [vocab[word] for word in words if vocab[word] > 0]
                word_num = len(idxs)
                label_data.append(idxs)
                itemids.append(ss[0])
                word_count += word_num

                jra = []

                jd_line = str(ss[2])
                jd_skill = jd_line.strip().split(',')
                jd_idxs = [vocab[word] for word in jd_skill if vocab[word] > 0]
                jra.extend(jd_idxs)

                resume_line = str(ss[3])
                resume_skill = resume_line.strip().split(',')
                resume_idxs = [vocab[word] for word in resume_skill if vocab[word] > 0]
                jra.extend(resume_idxs)

                ia1_line = str(ss[5])
                ia1_skill = ia1_line.strip().split(',')
                ia1_idxs = [vocab[word] for word in ia1_skill if vocab[word] > 0]
                #jra.extend(ia1_idxs)

                jd_resume_data.append(jra)

                split.append(int(ss[4]))

        else:
            for i, ss in enumerate(
                    csvdata[['index', round_data, 'jd_skill', 'resume_skill', 'train',
                             'assessment_skill2']].values):
                label_line = str(ss[1])
                words = label_line.strip().split(',')
                idxs = [vocab[word] for word in words if vocab[word] > 0]
                word_num = len(idxs)
                label_data.append(idxs)
                itemids.append(ss[0])
                word_count += word_num

                jra = []

                jd_line = str(ss[2])
                jd_skill = jd_line.strip().split(',')
                jd_idxs = [vocab[word] for word in jd_skill if vocab[word] > 0]
                jra.extend(jd_idxs)

                resume_line = str(ss[3])
                resume_skill = resume_line.strip().split(',')
                resume_idxs = [vocab[word] for word in resume_skill if vocab[word] > 0]
                jra.extend(resume_idxs)

                ia2_line = str(ss[5])
                ia2_skill = ia2_line.strip().split(',')
                ia1_idxs = [vocab[word] for word in ia2_skill if vocab[word] > 0]
                #jra.extend(ia1_idxs)

                jd_resume_data.append(jra)

                split.append(int(ss[4]))
        # print(label_data)
        # print(jd_resume_data)

        print('read corpus of round{} done!'.format(round))
        return label_data, jd_resume_data, vocab, word_count, itemids, split

    def pairVocab(self, threshold=0.05):
        pair_s = []
        pair_t = []
        n = self.ngram
        for sent in self.label_data:
            L = len(sent)
            nl = min(n + 1, L)
            for i in range(1, nl):
                pair = np.array([sent[:-i], sent[i:]]).transpose()
                tuple_pair = []
                for j in pair:
                    tuple_pair.append(tuple(j))
                pair_s.append(pair)
                pair_t.append(tuple_pair)
        pairs = np.concatenate(pair_s, axis=0)
        tmp = [tuple(t) for t in pairs]
        coun_dct = Counter(tmp)

        # tfidf_dic = {}  # tf-idf词典
        # for i in tmp:
        #   tfidf_dic[i] = tfidf(i, coun_dct, pair_t)

        self.pair_dct = {k: math.log(coun_dct[k]) for k in coun_dct if coun_dct[k] > threshold and k[0] != k[1]}
        # self.pair_dct = {k: tfidf_dic[k] for k in tfidf_dic if tfidf_dic[k] > threshold and k[0] != k[1]}
        #self.pair_dct = {k: math.log(coun_dct[k]) for k in coun_dct if coun_dct[k] > math.log(500) and k[0] != k[1]}
        sorted_key = sorted(self.pair_dct.keys(), key=lambda x: self.pair_dct[x], reverse=True)
        for i, key in enumerate(sorted_key):
            self.pair_dct[key] = i + 1  # start from 1
        self.whole_edge = np.array([k for k in sorted_key]).transpose()  ### 每一行为出度邻居a_{i,j} i->j
        # self.whole_edge_w = np.array([coun_dct[k] for k in sorted_key])
        self.whole_edge_w = np.array([math.log(coun_dct[k]) for k in sorted_key])
        print('pairVocab done!')
        print(self.whole_edge.shape)
        print(len(tmp))
        print(self.whole_edge.shape[1]/len(tmp))

    def process_jd_resume(self, sent):
        sent = sent
        idxs = np.unique(sent)
        idx_w_dict = Counter(sent)
        idx_w = []
        lens = 0
        for id in idxs:
            idx_w.append(idx_w_dict[id])
            lens += idx_w_dict[id]
        sidxs = []
        for id in sent:
            if id not in idxs and id not in sidxs:
                sidxs.append(id)
                idx_w.append(idx_w_dict[id])
                lens += idx_w_dict[id]
        if len(idxs) > 0 and len(sidxs) > 0:
            all_idxs = np.hstack([idxs, sidxs])
        elif len(idxs) == 0 and len(sidxs) > 0:
            all_idxs = np.array(sidxs)
        else:
            all_idxs = idxs
        # idxs.dtype=np.int
        assert lens == len(sent)

        return sent, all_idxs, idx_w

    def process_assessment(self, sent):
        n = self.ngram
        L = len(sent)
        pair_s = []
        edge_ids = []
        # for i in range(0, n-1):
        #     pair = np.array([sent[i:L-n+1+i], sent[n-1:L]]).transpose()
        nl = min(n + 1, L)
        for i in range(1, nl):
            pair = np.array([sent[:-i], sent[i:]]).transpose()
            pair_s.append(pair)
        pairs = np.concatenate(pair_s, axis=0)
        tmp = [tuple(t) for t in pairs]
        dct = Counter(tmp)
        dct1 = {k: math.log(dct[k]) for k in dct}
        keys = dct.keys()
        r, c, v = [], [], []
        for k in keys:
            try:
                edge_id = self.pair_dct[k]
            except:
                continue

            r.append(k[0])
            c.append(k[1])
            v.append(dct1[k])
            edge_ids.append(edge_id)
        # edge_index = np.array([c, r]) ### 每一行为入度邻居a_{i,j} j->i
        edge_index = np.array([r, c])  ### 每一行为出度邻居a_{i,j} i->j
        edge_w = np.array(v)
        idxs = np.unique(edge_index.reshape(-1))
        idx_w_dict = Counter(sent)
        idx_w = []
        lens = 0
        for id in idxs:
            idx_w.append(idx_w_dict[id])
            lens += idx_w_dict[id]
        sidxs = []
        for id in sent:
            if id not in idxs and id not in sidxs:
                sidxs.append(id)
                idx_w.append(idx_w_dict[id])
                lens += idx_w_dict[id]
        if len(idxs) > 0 and len(sidxs) > 0:
            all_idxs = np.hstack([idxs, sidxs])
        elif len(idxs) == 0 and len(sidxs) > 0:
            all_idxs = np.array(sidxs)
        else:
            all_idxs = idxs
        # idxs.dtype=np.int
        assert lens == len(sent)
        # if max(all_idxs)>10000:
        #     import ipdb
        #     ipdb.set_trace()
        if len(idxs) > 0:
            idxs_map = np.zeros(max(all_idxs) + 1)
            idxs_map[all_idxs] = range(len(all_idxs))
            edge_index = idxs_map[edge_index]
        else:
            edge_index = np.array([[], []])
        return sent, all_idxs, idx_w, edge_index, edge_w, edge_ids, L


class MyData(Data):
    def __init__(self, x=None, edge_w=None, edge_index=None, x_w=None, edge_id=None, text=None, text_w=None,
                 idx_sent=None, text_sent=None):
        super(MyData, self).__init__()
        if x is not None:
            self.x = x
        if edge_w is not None:
            self.edge_w = edge_w
        if edge_index is not None:
            self.edge_index = edge_index
        if x_w is not None:
            self.x_w = x_w
        if edge_id is not None:
            self.edge_id = edge_id
        if text is not None:
            self.text = text
        if text_w is not None:
            self.text_w = text_w
        if idx_sent is not None:
            self.idx_sent = idx_sent
        if text_sent is not None:
            self.text_sent = text_sent

    def __inc__(self, key, value):
        if 'index' in key or 'face' in key:
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value):
        if 'index' in key or 'face' in key:
            return 1
        elif key == 'x':
            return 0
        elif key == 'edge_id':
            return 0
        else:
            return 0


class MyDataset(InMemoryDataset):
    def __init__(self, root, dataset_path, vocab_path, ngram=3, vocab=None, transform=None, pre_transform=None,
                 STOPWORD=False,
                 edge_threshold=3, round='1'):
        self.rootPath = root
        self.vocab_path = vocab_path
        self.dataset_path = dataset_path
        head, tail = os.path.split(self.dataset_path)
        self.dataset_name = tail
        self.stop_str = '_stop' if STOPWORD else ''
        self.edge_threshold = edge_threshold
        if vocab is None:
            self.vocab = VocabEntry.from_corpus(self.vocab_path, withpad=True)
            # self.vocab.add(';')
        else:
            self.vocab = vocab
        self.ngram = ngram
        self.round = round
        super(MyDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.whole_edge, self.word_count, self.whole_edge_w = torch.load(
            self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        print('{}withoutia_round{}_ngram{}_threshold{}'.format(self.dataset_name,
                                       self.round,self.ngram,self.edge_threshold))
        # return [self.rootPath + '/graph_nragm%d_dataset%s.pt' % (self.ngram, self.stop_str)]
        return ['{}withoutia_round{}_ngram{}_threshold{}.pt'.format(self.dataset_name,
                                       self.round,self.ngram,self.edge_threshold)]  # data1.pt(exp_data.tsv,labels)#data3.pt(exp_data3.tsv,final_dic) data6.pt <pad>=0

    def download(self):
        pass

    def process(self):
        dataset = PreTextData(self.dataset_path, ngram=self.ngram, vocab=self.vocab,
                              edge_threshold=self.edge_threshold,
                              round=self.round)  # TODO important parameter for different datasets
        data_list = []
        used_list = []
        for i in range(len(dataset)):

            label_sent = dataset.label_data[i]
            # print('label_sent', label_sent)
            # predict2 = [vocab.id2word(int(j)) for j in label_sent]
            # print(predict2)

            jd_resume_sent = dataset.jd_resume_data[i]
            # print('jd_resume_sent', jd_resume_sent)
            # predict3 = [vocab.id2word(int(j)) for j in jd_resume_sent]
            # print(predict3)
            split_train = dataset.split[i]


            if len(label_sent) > 1:
                idx_sent, idxs, idx_w, edge_index, edge_w, edge_id, L = dataset.process_assessment(
                    label_sent)  # 构成图的技能点
                text_sent, text_indxs, text_w = dataset.process_jd_resume(jd_resume_sent)

            if edge_index.shape[1] >= 0:
                used_list.append(dataset.itemids[i])
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                x = torch.tensor(idxs, dtype=torch.long)
                idx_w = torch.tensor(idx_w, dtype=torch.float)
                edge_w = torch.tensor(edge_w, dtype=torch.float)
                edge_id = torch.tensor(edge_id, dtype=torch.long)
                idx_sent = torch.tensor(idx_sent, dtype=torch.long)
                text_sent = torch.tensor(text_sent, dtype=torch.long)
                text = torch.tensor(text_indxs, dtype=torch.long)
                text_w = torch.tensor(text_w, dtype=torch.float)
                split = torch.tensor(split_train, dtype=torch.long).unsqueeze(0)

                d = MyData(x=x, edge_w=edge_w, edge_index=edge_index,
                           x_w=idx_w, edge_id=edge_id, text=text, text_w=text_w, idx_sent=idx_sent, text_sent=text_sent)
                d.split = split

                data_list.append(d)
        np.save(self.rootPath + '/withoutia_used_{}_round{}_ngram{}_threshold{}'.format(self.dataset_name, self.round,self.ngram,self.edge_threshold), used_list)
        data, slices = self.collate(data_list)
        torch.save((data, slices, dataset.whole_edge, dataset.word_count, dataset.whole_edge_w),
                   self.processed_paths[0])
import codecs
import json
from os.path import join
import pickle
import os
import re
import numpy as np


################# Load and Save Data ################
from gensim.models import word2vec


def load_json(rfdir, rfname):
    with codecs.open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        return json.load(rf)


def dump_json(obj, wfpath, wfname, indent=None):
    with codecs.open(join(wfpath, wfname), 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, ensure_ascii=False, indent=indent)


def dump_data(obj, wfpath, wfname):
    with open(os.path.join(wfpath, wfname), 'wb') as wf:
        pickle.dump(obj, wf)


def load_data(rfpath, rfname):
    with open(os.path.join(rfpath, rfname), 'rb') as rf:
        return pickle.load(rf)


################# Random Walk ################

import random


class MetaPathGenerator:
    def __init__(self):
        #生成异质网络保存以下键值对
        self.paper_author = dict()
        self.author_paper = dict()
        self.paper_org = dict()
        self.org_paper = dict()
        self.paper_conf = dict()
        self.conf_paper = dict()

    def read_data(self, dirpath):
        temp = set()

        with open(dirpath + "/paper_org.txt", encoding='utf-8') as pafile:
            for line in pafile:
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in self.paper_org:
                    self.paper_org[p] = []
                self.paper_org[p].append(a)
                if a not in self.org_paper:
                    self.org_paper[a] = []
                self.org_paper[a].append(p)
        temp.clear()

        with open(dirpath + "/paper_author.txt", encoding='utf-8') as pafile:
            for line in pafile:
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in self.paper_author:
                    self.paper_author[p] = []
                self.paper_author[p].append(a)
                if a not in self.author_paper:
                    self.author_paper[a] = []
                self.author_paper[a].append(p)
        temp.clear()

        with open(dirpath + "/paper_conf.txt", encoding='utf-8') as pcfile:
            for line in pcfile:
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in self.paper_conf:
                    self.paper_conf[p] = []
                self.paper_conf[p].append(a)
                if a not in self.conf_paper:
                    self.conf_paper[a] = []
                self.conf_paper[a].append(p)
        temp.clear()

        print("#papers ", len(self.paper_conf))
        print("#authors", len(self.author_paper))
        print("#org_words", len(self.org_paper))
        print("#confs  ", len(self.conf_paper))

    def generate_WMRW(self, outfilename, numwalks, walklength):
        outfile = open(outfilename, 'w')
        for paper0 in self.paper_conf:
            for j in range(0, numwalks):  # wnum walks 随机游走轮数
                paper = paper0
                outline = ""
                i = 0
                while (i < walklength): #walklength 随机游走路径长
                    i = i + 1
                    # paper->CoAuthor->paper
                    if paper in self.paper_author:
                        authors = self.paper_author[paper]  #获取该论文下的所有作者
                        numa = len(authors)
                        authorid = random.randrange(numa)   #随机选取一个author来搜索下一个paper
                        author = authors[authorid]

                        papers = self.author_paper[author]
                        nump = len(papers)
                        if nump > 1:
                            paperid = random.randrange(nump)
                            paper1 = papers[paperid]
                            while paper1 == paper:
                                paperid = random.randrange(nump)
                                paper1 = papers[paperid]
                            paper = paper1
                            outline += " " + paper
                    # paper->CoOrg->paper
                    if paper in self.paper_org:
                        words = self.paper_org[paper]
                        numw = len(words)
                        wordid = random.randrange(numw)   #随机选取一个word来搜索下一个paper
                        word = words[wordid]

                        papers = self.org_paper[word]
                        nump = len(papers)
                        if nump > 1:
                            paperid = random.randrange(nump)
                            paper1 = papers[paperid]
                            while paper1 == paper:
                                paperid = random.randrange(nump)
                                paper1 = papers[paperid]
                            paper = paper1
                            outline += " " + paper

                outfile.write(outline + "\n")
        outfile.close()

        print("walks done")


################# Compare Lists ################
# 利用tanimoto比较两篇论文各类list的相似度度量
def tanimoto(p, q):
    c = [v for v in p if v in q]
    return float(len(c) / (len(p) + len(q) - len(c)))


################# Paper similarity ################

def generate_pair(pubs, outlier):  ##求匹配相似度
    dirpath = 'gene'

    paper_org = {}
    paper_conf = {}
    paper_author = {}
    paper_word = {}

    temp = set()
    with open(dirpath + "/paper_org.txt", encoding='utf-8') as pafile:
        for line in pafile:
            temp.add(line)
    for line in temp:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            p, a = toks[0], toks[1]
            if p not in paper_org:
                paper_org[p] = []
            paper_org[p].append(a)
    temp.clear()

    with open(dirpath + "/paper_conf.txt", encoding='utf-8') as pafile:
        for line in pafile:
            temp.add(line)
    for line in temp:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            p, a = toks[0], toks[1]
            if p not in paper_conf:
                paper_conf[p] = []
            paper_conf[p] = a
    temp.clear()

    with open(dirpath + "/paper_author.txt", encoding='utf-8') as pafile:
        for line in pafile:
            temp.add(line)
    for line in temp:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            p, a = toks[0], toks[1]
            if p not in paper_author:
                paper_author[p] = []
            paper_author[p].append(a)
    temp.clear()

    with open(dirpath + "/paper_word.txt", encoding='utf-8') as pafile:
        for line in pafile:
            temp.add(line)
    for line in temp:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            p, a = toks[0], toks[1]
            if p not in paper_word:
                paper_word[p] = []
            paper_word[p].append(a)
    temp.clear()

    paper_paper = np.zeros((len(pubs), len(pubs)))
    for i, pid in enumerate(pubs):
        if i not in outlier:
            continue
        for j, pjd in enumerate(pubs):
            if j == i:
                continue
            ca = 0
            cv = 0
            co = 0
            ct = 0

            if pid in paper_author and pjd in paper_author:
                ca = len(set(paper_author[pid]) & set(paper_author[pjd])) * 1.5
            if pid in paper_conf and pjd in paper_conf and 'null' not in paper_conf[pid]:
                cv = tanimoto(set(paper_conf[pid]), set(paper_conf[pjd]))
            if pid in paper_org and pjd in paper_org:
                co = tanimoto(set(paper_org[pid]), set(paper_org[pjd]))
            if pid in paper_word and pjd in paper_word:
                ct = len(set(paper_word[pid]) & set(paper_word[pjd])) / 3

            paper_paper[i][j] = ca + cv + co + ct

    return paper_paper


################# Evaluate ################

def pairwise_evaluate(correct_labels, pred_labels):
    TP = 0.0  # Pairs Correctly Predicted To SameAuthor
    TP_FP = 0.0  # Total Pairs Predicted To SameAuthor
    TP_FN = 0.0  # Total Pairs To SameAuthor

    for i in range(len(correct_labels)):
        for j in range(i + 1, len(correct_labels)):
            if correct_labels[i] == correct_labels[j]:
                TP_FN += 1
            if pred_labels[i] == pred_labels[j]:
                TP_FP += 1
            if (correct_labels[i] == correct_labels[j]) and (pred_labels[i] == pred_labels[j]):
                TP += 1

    if TP == 0:
        pairwise_precision = 0
        pairwise_recall = 0
        pairwise_f1 = 0
    else:
        pairwise_precision = TP / TP_FP
        pairwise_recall = TP / TP_FN
        pairwise_f1 = (2 * pairwise_precision * pairwise_recall) / (pairwise_precision + pairwise_recall)
    return pairwise_precision, pairwise_recall, pairwise_f1


################# Save Paper Features ################

def save_relation(name_pubs_raw, name):  # 保存论文的各种feature
    name_pubs_raw = load_json('genename', name_pubs_raw)
    ## trained by all text in the datasets. Training code is in the cells of "train word2vec"
    save_model_name = "word2vec/Aword2vec.model"
    model_w = word2vec.Word2Vec.load(save_model_name)

    r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
    stopword = ['at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with', 'the', 'by', 'we', 'be',
                'is', 'are', 'can']
    stopword1 = ['university', 'univ', 'china', 'department', 'dept', 'laboratory', 'lab', 'school', 'al', 'et',
                 'institute', 'inst', 'college', 'chinese', 'beijing', 'journal', 'science', 'international']

    f1 = open('gene/paper_author.txt', 'w', encoding='utf-8')
    f2 = open('gene/paper_conf.txt', 'w', encoding='utf-8')
    f3 = open('gene/paper_word.txt', 'w', encoding='utf-8')
    f4 = open('gene/paper_org.txt', 'w', encoding='utf-8')

    taken = name.split("_")#把名字从_分成两部分
    name = taken[0] + taken[1]
    name_reverse = taken[1] + taken[0]
    if len(taken) > 2:
        name = taken[0] + taken[1] + taken[2]
        name_reverse = taken[2] + taken[0] + taken[1]

    authorname_dict = {}
    ptext_emb = {}

    tcp = set()
    for i, pid in enumerate(name_pubs_raw):

        pub = name_pubs_raw[pid]

        # save authors
        org = ""
        for author in pub["authors"]:
            authorname = re.sub(r, '', author["name"]).lower()
            taken = authorname.split(" ")
            if len(taken) == 2:  ##检测目前作者名是否在作者词典中
                authorname = taken[0] + taken[1]
                authorname_reverse = taken[1] + taken[0]

                if authorname not in authorname_dict:
                    if authorname_reverse not in authorname_dict:
                        authorname_dict[authorname] = 1
                    else:
                        authorname = authorname_reverse
            else:
                authorname = authorname.replace(" ", "")

            if authorname != name and authorname != name_reverse:
                f1.write(pid + '\t' + authorname + '\n')

            else:
                if "org" in author: #只取当前作为候选者名字的机构
                    org = author["org"]

        # save org 待消歧作者的机构名
        pstr = org.strip()
        pstr = pstr.lower()  # 小写
        pstr = re.sub(r, ' ', pstr)  # 去除符号
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()  # 去除多余空格
        pstr = pstr.split(' ')
        pstr = [word for word in pstr if len(word) > 1]
        pstr = [word for word in pstr if word not in stopword1]
        pstr = [word for word in pstr if word not in stopword]
        pstr = set(pstr)
        for word in pstr:
            f4.write(pid + '\t' + word + '\n')

        # save venue
        # 保存会议地点（名称）
        pstr = pub["venue"].strip()
        pstr = pstr.lower()
        pstr = re.sub(r, ' ', pstr)
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
        pstr = pstr.split(' ')
        pstr = [word for word in pstr if len(word) > 1]
        pstr = [word for word in pstr if word not in stopword1]
        pstr = [word for word in pstr if word not in stopword]
        for word in pstr:
            f2.write(pid + '\t' + word + '\n')
        if len(pstr) == 0:
            f2.write(pid + '\t' + 'null' + '\n')

        # save text
        pstr = ""
        keyword = ""
        if "keywords" in pub:
            for word in pub["keywords"]:
                keyword = keyword + word + " "
        pstr = pstr + pub["title"]
        pstr = pstr.strip()
        pstr = pstr.lower()
        pstr = re.sub(r, ' ', pstr)
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
        pstr = pstr.split(' ')
        pstr = [word for word in pstr if len(word) > 1]
        pstr = [word for word in pstr if word not in stopword]
        for word in pstr:
            f3.write(pid + '\t' + word + '\n')

        # save all words' embedding
        # 将关键词、标题、地点、org机构、year利用训练词向量模型embedding为100维向量
        pstr = keyword + " " + pub["title"] + " " + pub["venue"] + " " + org
        if "year" in pub:
            pstr = pstr + " " + str(pub["year"])
        pstr = pstr.strip()
        pstr = pstr.lower()
        pstr = re.sub(r, ' ', pstr)
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
        pstr = pstr.split(' ')
        pstr = [word for word in pstr if len(word) > 2]
        pstr = [word for word in pstr if word not in stopword]
        pstr = [word for word in pstr if word not in stopword1]

        words_vec = []
        for word in pstr:
            if (word in model_w):
                words_vec.append(model_w[word])
        if len(words_vec) < 1:
            words_vec.append(np.zeros(100))
            tcp.add(i)
            # print ('outlier:',pid,pstr)
        ptext_emb[pid] = np.mean(words_vec, 0)

    #  ptext_emb: key is paper id, and the value is the paper's text embedding
    dump_data(ptext_emb, 'gene', 'ptext_emb.pkl')
    # the paper index that lack text information
    dump_data(tcp, 'gene', 'tcp.pkl')

    f1.close()
    f2.close()
    f3.close()
    f4.close()
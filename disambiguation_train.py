import utils

pubs_raw = utils.load_json("train","train_pub.json")#每个名字中所有论文的原始信息
name_pubs = utils.load_json("train","test_rnn.json")#标签，一个名字分成多少簇

import re
from gensim.models import word2vec
from sklearn.cluster import DBSCAN,AgglomerativeClustering
import numpy as np
from keras.models import load_model
from sklearn.metrics.pairwise import pairwise_distances
from count import root_mean_log_squared_error,root_mean_squared_error
result = []
filename = 'train_avgf1.txt'
with open(filename, 'w') as file_object:
    file_object.write("all_name_f1:\n")

for n, name in enumerate(name_pubs): #对每一个候选集名字进行操作
    ilabel = 0
    pubs = []  # all papers
    labels = []  # ground truth
# 保存真实的同名作者论文分类
    for author in name_pubs[name]:#每一簇，一个作者，一个人
        iauthor_pubs = name_pubs[name][author]#得到同一个人的所有论文
        for pub in iauthor_pubs:
            pubs.append(pub)
            labels.append(ilabel)
        #同作者名下不同人的论文打上不同标签
        ilabel += 1

    print(n, name, len(pubs))#该候选集的论文数量

    if len(pubs) == 0:
        result.append(0)
        continue

    ##保存关系
    ###############################################################
    name_pubs_raw = {}
    for i, pid in enumerate(pubs):#去把论文的原始信息取回来
        name_pubs_raw[pid] = pubs_raw[pid]

    utils.dump_json(name_pubs_raw, 'genename', name + '.json', indent=4) #该候选集的所有文章详细信息保存
    utils.save_relation(name + '.json', name)
    ###############################################################

    ##元路径游走类
    ###############################################################r
    mpg = utils.MetaPathGenerator()
    mpg.read_data("gene")
    ###############################################################

    ##论文关系表征向量
    ###############################################################
    all_embs = []
    rw_num = 3
    cp = set()  #离散论文集
    for k in range(rw_num):
        mpg.generate_WMRW("gene/RW.txt", 5, 20)  # 生成路径集
        sentences = word2vec.Text8Corpus(r'gene/RW.txt') #分析的语料，从文件中遍历读出，利用Text8Corpus构建
        #size 词向量的维度 min_count: 对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5。
        #window：即词向量上下文最大距离，基于滑动窗口来做预测。
        model = word2vec.Word2Vec(sentences, size=100, negative=25, min_count=1, window=10)
        embs = []
        for i, pid in enumerate(pubs):
            if pid in model:
                embs.append(model[pid])
            else:
                cp.add(i)               #保存孤立节点至离群论文集中
                embs.append(np.zeros(100))  #关系表征向量设为全0向量
        all_embs.append(embs)
    all_embs = np.array(all_embs)
    print('relational outlier:', cp)
    ###############################################################

    ##论文文本表征向量
    ###############################################################
    ptext_emb = utils.load_data('gene', 'ptext_emb.pkl')
    tcp = utils.load_data('gene', 'tcp.pkl')
    print('semantic outlier:', tcp)
    tembs = []
    for i, pid in enumerate(pubs):
        tembs.append(ptext_emb[pid])
    ###############################################################

    ##离散点
    outlier = set()
    for i in cp:
        outlier.add(i)
    for i in tcp:
        outlier.add(i)

    ##网络嵌入向量相似度
    #每组论文向量求余弦相似性矩阵，再对这k个相似性矩阵求均值，得到最终的论文关系相似性矩阵
    sk_sim = np.zeros((len(pubs), len(pubs)))
    for k in range(rw_num):
        sk_sim = sk_sim + pairwise_distances(all_embs[k], metric="cosine")
    sk_sim = sk_sim / rw_num

    ##文本相似度
    t_sim = pairwise_distances(tembs, metric="cosine")

    w = 1
    sim = (np.array(sk_sim) + w * np.array(t_sim)) / (1 + w)

    ##evaluate
    ###############################################################
    
    sample=[]
    x=[]
    sampled_points = [pubs[p] for p in np.random.choice(len(pubs), 300, replace=True)]
    for p in sampled_points:
        x.append(ptext_emb[p])
    sample.append(np.stack(x))
    sample = np.stack(sample)
    rnn_model = load_model('rnn.h5',custom_objects={'root_mean_squared_error':root_mean_squared_error, 'root_mean_log_squared_error': root_mean_log_squared_error})
    pre_num = rnn_model.predict(sample).astype(np.int16)
    #pre = DBSCAN(eps=0.2, min_samples=4, metric="precomputed").fit_predict(sim)
    pre = AgglomerativeClustering(n_clusters=pre_num[0,0],affinity='precomputed',linkage='complete').fit_predict(sim)
    for i in range(len(pre)):
        if pre[i] == -1:
            outlier.add(i)

    ## assign each outlier a label
    paper_pair = utils.generate_pair(pubs, outlier)
    paper_pair1 = paper_pair.copy()
    K = len(set(pre))
    for i in range(len(pre)):
        if i not in outlier:
            continue
        j = np.argmax(paper_pair[i])
        while j in outlier:
            paper_pair[i][j] = -1
            j = np.argmax(paper_pair[i])
        if paper_pair[i][j] >= 1.5:
            pre[i] = pre[j]
        else:
            pre[i] = K
            K = K + 1

    ## find nodes in outlier is the same label or not
    for ii, i in enumerate(outlier):
        for jj, j in enumerate(outlier):
            if jj <= ii:
                continue
            else:
                if paper_pair1[i][j] >= 1.5:
                    pre[j] = pre[i]

    labels = np.array(labels)
    pre = np.array(pre)
    print(labels, len(set(labels)))
    print(pre, len(set(pre)))
    pairwise_precision, pairwise_recall, pairwise_f1 = utils.pairwise_evaluate(labels, pre)
    print(pairwise_precision, pairwise_recall, pairwise_f1)
    result.append(pairwise_f1)


    with open(filename, 'a') as file_object:
        file_object.write(str(pairwise_f1)+"\n")

with open(filename, 'a') as file_object:
    file_object.write("\n"+str(np.mean(result))+"\n")
print('avg_f1:', np.mean(result))
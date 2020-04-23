import utils
import re
import os
import codecs
import pickle
import json
import numpy as np
from gensim.models import word2vec
#把所有文章的特征向量保存下来

pubs_raw = utils.load_json("train","train_pub.json")#每个名字中所有论文的原始信息
name_pubs = utils.load_json("train","train_author.json")#标签，一个名字分成多少簇
save_model_name = "word2vec/Aword2vec.model"
model_w = word2vec.Word2Vec.load(save_model_name)

r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
stopword = ['at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with', 'the', 'by', 'we', 'be',
                'is', 'are', 'can']
stopword1 = ['university', 'univ', 'china', 'department', 'dept', 'laboratory', 'lab', 'school', 'al', 'et',
                 'institute', 'inst', 'college', 'chinese', 'beijing', 'journal', 'science', 'international']
ptext_emb = {}
authorname_dict = {}
for n, name in enumerate(name_pubs):
    print(n)
    taken = name.split("_")#把名字从_分成两部分
    name1 = taken[0] + taken[1]
    name1_reverse = taken[1] + taken[0]
    if len(taken) > 2:
        name1 = taken[0] + taken[1] + taken[2]
        name1_reverse = taken[2] + taken[0] + taken[1]

    for author in name_pubs[name]:#每一簇，一个作者，一个人
        iauthor_pubs = name_pubs[name][author]#得到同一个人的所有论文
        for pid in iauthor_pubs:
            pub = pubs_raw[pid]

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

                if authorname != name and authorname != name1_reverse:
                    continue
                else:
                    if "org" in author: #只取当前作为候选者名字的机构
                        org = author["org"]

            keyword = ""
            if "keywords" in pub:
                for word in pub["keywords"]:
                    keyword = keyword + word + " "

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
            ptext_emb[pid] = np.mean(words_vec, 0)

utils.dump_data(ptext_emb, 'gene', 'ptext_emb_all.pkl')                

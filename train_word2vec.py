## SAVE all text in the datasets

import codecs
import json
from os.path import join
import pickle
import os
import re
import utils

#preprocessing

pubs_raw = utils.load_json("train", "train_pub.json")

r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
f1 = open('gene/all_text.txt', 'w', encoding='utf-8')
length = len(pubs_raw)
for i, pid in enumerate(pubs_raw):
    if i%10000==0:
        print("%d/%d",i,length)
    pub = pubs_raw[pid]

    for author in pub["authors"]:
        if "org" in author:
            org = author["org"]
            pstr = org.strip()
            pstr = pstr.lower()
            pstr = re.sub(r, ' ', pstr)
            pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
            f1.write(pstr + '\n')

    title = pub["title"]
    pstr = title.strip()
    pstr = pstr.lower()
    pstr = re.sub(r, ' ', pstr)
    pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
    f1.write(pstr + '\n')

    if "abstract" in pub and type(pub["abstract"]) is str:
        abstract = pub["abstract"]
        pstr = abstract.strip()
        pstr = pstr.lower()
        pstr = re.sub(r, ' ', pstr)
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
        f1.write(pstr + '\n')

    venue = pub["venue"]
    pstr = venue.strip()
    pstr = pstr.lower()
    pstr = re.sub(r, ' ', pstr)
    pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
    f1.write(pstr + '\n')

f1.close()


from gensim.models import word2vec

sentences = word2vec.Text8Corpus(r'gene/all_text.txt')
model = word2vec.Word2Vec(sentences, size=100,negative =5, min_count=2, window=5)
model.save('word2vec/Aword2vec.model')

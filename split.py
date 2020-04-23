
import numpy as np
import utils
import pickle
import matplotlib.pyplot as plt

#数据集分成两个部分
train_author={}
test_author={}
name_pubs = utils.load_json("train","train_author.json")
for n, name in enumerate(name_pubs): 
    if n<150:
        train_author[name]=name_pubs[name]
    else:
        test_author[name]=name_pubs[name]
utils.dump_json(train_author,"train","train_rnn.json")
utils.dump_json(test_author,"train","test_rnn.json")

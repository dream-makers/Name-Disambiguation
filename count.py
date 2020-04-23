from os.path import join
import numpy as np
import keras.backend as K
import tensorflow as tf
import os
import codecs
import pickle
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
'''
改进：
1.预测的时候，对每一个候选集，应该抽取多次进行预测取一个平均值
2.改变训练数据集的选取

'''
def load_data(rfpath, rfname):
    with open(os.path.join(rfpath, rfname), 'rb') as rf:
        return pickle.load(rf)

def load_json(rfdir, rfname):
    with codecs.open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        return json.load(rf)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def root_mean_log_squared_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), np.inf) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), np.inf) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))


def create_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(64), input_shape=(300, 100)))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss="msle",
                  optimizer='rmsprop',
                  metrics=[root_mean_squared_error, "accuracy", "msle", root_mean_log_squared_error])

    return model


def sampler(clusters, k=300, batch_size=10, min=1, max=300, flatten=False):
    #clusters是所有供训练的簇的合集
    xs, ys = [], []
    for b in range(batch_size):#为每一个batch生成训练数据
        #生成此次的标签
        #num_clusters = np.random.randint(min, max)
        num_clusters = b%max+1
        #随机选择num_clusters个簇
        sampled_clusters = np.random.choice(len(clusters),num_clusters , replace=False)
        #把训练的簇中的所有文章放到items中
        items = []
        for c in sampled_clusters:
            items.extend(clusters[c])#extend是把集合放进去
        #摘出k=300个文章
        sampled_points = [items[p] for p in np.random.choice(len(items), k, replace=True)]
        x = []
        for p in sampled_points:
            x.append(data_cache[p])
        if flatten:
            xs.append(np.sum(x, axis=0))
        else:
            xs.append(np.stack(x))#xs存储每一个batch的数据
        ys.append(num_clusters)
    return np.stack(xs), np.stack(ys)


def gen_train(clusters, k=300, batch_size=1000, flatten=False):
    while True:
        yield sampler(clusters, k, batch_size, flatten=flatten)


def gen_test(k=300, flatten=False):
    name_to_pubs_test = load_json("train","test_rnn.json")#标签，一个名字分成多少簇
    xs, ys = [], []
    names = []
    for n,name in enumerate(name_to_pubs_test):
        
        num_clusters = len(name_to_pubs_test[name])
        items = []#存储候选集的所有文章
        for c in name_to_pubs_test[name]:  # one person
            for item in name_to_pubs_test[name][c]:
                items.append(item)
        #对每个候选集计算3次再取平均值
        for i in range(3):
            x = []
            #从此名字的所有文章中抽取k=300个文章
            sampled_points = [items[p] for p in np.random.choice(len(items), k, replace=True)]
            for p in sampled_points:
                x.append(data_cache[p])
            if flatten:
                xs.append(np.sum(x, axis=0))
            else:
                xs.append(np.stack(x))
            ys.append(num_clusters)
            names.append(name)
    xs = np.stack(xs)
    ys = np.stack(ys)
    return names, xs, ys


def run_rnn(k=300, seed=1106):
    
    name_pubs= load_json("train","train_rnn.json")#标签，一个名字分成多少簇
    test_names, test_x, test_y = gen_test(k)
    np.random.seed(seed)
    clusters = []
    #生成训练使用的簇的集合
    for n, name in enumerate(name_pubs):
        for author in name_pubs[name]:#每一簇，一个作者，一个人
            clusters.append(name_pubs[name][author])
    '''
    for i, c in enumerate(clusters):
        if i % 100 == 0:
            print(i, len(c), len(clusters))
        for pid in c:
            data_cache[pid] = lc.get(pid) #先取出来，训练的时候直接从缓存里找
    '''
    model = create_model()
    # print(model.summary())
    history=model.fit_generator(gen_train(clusters, k=300, batch_size=600), steps_per_epoch=100, epochs=1000,
                        validation_data=(test_x, test_y))
    kk = model.predict(test_x)
    wf = open(join('gene', 'n_clusters_rnn.txt'), 'w')
    for i, name in enumerate(test_names):
        if (i+1)%3==0:
            avg =(kk[i][0]+kk[i-1][0]+ kk[i-2][0])/3
            wf.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(name, test_y[i], kk[i][0], kk[i-1][0], kk[i-2][0], avg))
    wf.close()
    model.save('rnn.h5')
    with open('trainlog.txt','wb') as wf:
        pickle.dump(history.history,wf)


if __name__ == '__main__':
    data_cache = load_data('gene', 'ptext_emb_all.pkl')
    run_rnn()

# Name-Disambiguation
执行顺序
1.运行train_word2vec.py,利用原始数据训练word2vec模型
2.运行prepare.py,生成并保存每篇文章的文本特征向量
3.运行count.py,训练预测簇大小的模型
4.运行disambiguation_train.py,对每个候选集进行消歧

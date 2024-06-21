#!/usr/bin/env python
# -*- encoding: utf-8 -*-


# ML_MODEL_NAME = ["lg", "knn", "dt", "rf", "gbdt", "xgb", "catboost", "svm", "bayes"]

# DL_MODEL_NAME = ["lstm", "cnn", "transformer", "capsule"]

ML_MODEL_NAME = ["lg", "knn", "gbdt", "svm"]

DL_MODEL_NAME = ["lstm", "cnn"]

BATCH_SIZE = 8

SPLIT_SIZE = 0.3

IS_SAMPLE = True


VOCAB_MAX_SIZE = 100000  # 词表中词的最大数量

WORD_MIN_FREQ = 5  # 词表中一个单词出现的最小频率

VOCAB_SAVE_PATH = "./data/vocab_dic.pkl"  # 词表存储的位置

L2I_SAVE_PATH = "./data/label2id.pkl"  #  label的映射表

PRETRAIN_EMBEDDING_FILE = "./data/embed.txt"

VERBOSE = 1  # 每隔10个epoch 输出一次训练结果和测试的loss

MAX_SEQ_LEN = 100  # 使用预训练模型时，设置允许每条文本数据的最长长度

DEVICE = 1  # 存在GPU时指定GPU卡号(单卡,包括训练和预测)


KEYWORD_SIMILAR_WORD2VEC = "data/word2vec/embedding/tencent-embedding-zh-d100-append-zhihu.bin"  # 关键词相似度词向量文件

BGE_SMALL_MODEL = "save_model/bge-small-zh-v1.5"

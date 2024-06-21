#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:haobingchaun
@desc:词向量训练、词向量增量训练、词向量获取
@time:2024/03/15
"""

from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import numpy as np


class TrainWord2Vec:
    def __init__(self, num_features=50, min_word_count=1, context=5):
        """
        构造词向量训练类，初始化词向量训练参数
        :param
            num_features:  返回的向量长度
            min_word_count:  最低词频
            context: 滑动窗口大小
        """
        self.num_features = num_features
        self.min_word_count = min_word_count
        self.context = context

    def get_text(self, text_file: str):
        corpus = []
        # 打开分词后的文本
        for line in open(text_file, "r", encoding="utf-8"):
            text = line.strip().split(" ")
            corpus.append(text)
        # print(corpus)
        return corpus

    def train_model(self, text_file: str, save_path: str):
        """
        从头训练word2vec模型
        :param
            text_file: 切词后的语料数据文件
            save_path : 存储词向量模型路径
        :return: word2vec模型
        """
        text = self.get_text(text_file)
        model = Word2Vec(
            sentences=text,
            vector_size=self.num_features,
            min_count=self.min_word_count,
            window=self.context,
        )
        save_model = save_path.split(".")[-1]
        if save_model not in ["model", "wordvectors", "txt", "bin"]:
            print("文件格式错误，请重新输入")
            # 保存为Word2Vec对象，包含完整的参数、词汇频率和二叉树信息
        if save_model == "model":
            model.save(save_path)
        # 以下保存为 “word2vec C format”对象，不包含完整的参数、词汇频率和二叉树信息，只剩下词向量信息
        elif save_model == "wordvectors":
            model.wv.save(save_path)
        elif save_model == "txt":
            model.wv.save_word2vec_format(save_path, binary=False)
        elif save_model == "bin":
            model.wv.save_word2vec_format(save_path, binary=True)

    def update_model(self, text_file, old_model, old_model_size, new_model_path):
        """
        对已完成训练的model/txt格式文件进行增量训练
        :param
        text_file: 经过清洗之后的新的语料数据
        old_model: 已完成训练的词向量文件
        old_model_size: 已完成训练词向量文件词向量维度
        new_model_path：增量训练词向量文件路径
        :return: word2vec模型
        """
        text = self.get_text(text_file)
        assert new_model_path.split(".")[-1] == "bin"
        try:
            model_format = old_model.split(".")[-1]
            if model_format == "model":
                model = Word2Vec.load(
                    old_model
                )  # 加载旧模型,用此种加载方式(完整信息的Word2Vec对象)可以进行增量训练
                model.build_vocab(text, update=True)  # 更新词汇表
                model.train(
                    corpus_iterable=text, total_examples=model.corpus_count, epochs=1
                )  # epoch=iter语料库的迭代次数；（默认为5）  total_examples:句子数。
                model.wv.save_word2vec_format(
                    new_model_path, binary=True
                )  # 默认格式保存为bin,只保留词向量信息
            # 加载方式(不需要完整信息的Word2Vec对象)便可以可以进行增量训练,强
            elif model_format == "txt":
                # 首先初始化一个word2vec 模型：
                w2v_model = Word2Vec(vector_size=old_model_size, sg=1, min_count=0)
                w2v_model.build_vocab(text)
                # 再加载第三方预训练模型：
                third_model = KeyedVectors.load_word2vec_format(old_model, binary=False)
                # 通过 intersect_word2vec_format()方法merge词向量：
                w2v_model.build_vocab(
                    [list(third_model.key_to_index.keys())], update=True
                )
                w2v_model.wv.vectors_lockf = np.ones(
                    len(w2v_model.wv), dtype=np.float32
                )
                w2v_model.wv.intersect_word2vec_format(
                    old_model, binary=False, lockf=1.0
                )
                w2v_model.train(
                    text, total_examples=w2v_model.corpus_count, epochs=w2v_model.epochs
                )
                w2v_model.wv.save_word2vec_format(
                    new_model_path, binary=True
                )  # 默认格式保存为bin,只保留词向量信息
        except Exception as e:
            print("词向量模型格式不对")
            print(e)

    def get_word_vector(self, model, word: str):
        """
        获取指定单词词向量
        param:
        model:词向量文件
        word:待测试单词
        """
        model_format = model.split(".")[-1]
        if model_format == "model":
            model = Word2Vec.load(model)  # 加载旧模型，包含完整对象信息model格式文件
            print(model.wv[word])
        elif model_format == "txt":
            model = KeyedVectors.load_word2vec_format(
                model, binary=False
            )  # 加载旧模型，只包含词向量信息,txt格式
            print(model[word])
        elif model_format == "bin":
            model = KeyedVectors.load_word2vec_format(
                model, binary=True
            )  # 加载旧模型，只包含词向量信息,bin格式
            print(model[word])
        elif model_format == "wordvectors":
            model = KeyedVectors.load(model)  # 加载旧模型，只包含词向量信息
            print(model[word])


if __name__ == "__main__":
    train_model = TrainWord2Vec()
    train_model.train_model(
        "data/word2vec/text/zhihu.txt",
        save_path="data/word2vec/embedding/zhihu_embedding.txt",
    )
    train_model.update_model(
        "data/word2vec/text/zhihu.txt",
        old_model="data/word2vec/embedding/zhihu_embedding.txt",
        old_model_size=50,
        new_model_path="data/word2vec/embedding/zhihu_embedding_append.bin",
    )
    # train_model.get_word_vector(
    #     "data/word2vec/embedding/zhihu_embedding_append.bin", "人工智能"
    # )

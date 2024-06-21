"""
根据预定义关键词计算文本中关键词的相似度
"""

import sys
import os

sys.path.append(os.getcwd())
from data.text_representation import cutsentences
from gensim.models import KeyedVectors
from math import sqrt
import time
from pprint import pprint
from logConfig import logger


class WordSimilarity:
    def __init__(self, model):
        self.result_total_antonym = []
        self.result_total_similar = []
        start = time.time()
        self.model = KeyedVectors.load_word2vec_format(model, binary=True)
        logger.info("模型加载完成,耗时" + str(time.time() - start) + "秒")

    def square_rooted(self, x):
        return round(sqrt(sum([a * a for a in x])), 6)

    def cosine_similarity(self, x, y):
        numerator = sum(a * b for a, b in zip(x, y))
        denominator = self.square_rooted(x) * self.square_rooted(y)
        return round(numerator / float(denominator), 6)

    def get_word_similarity(self, word1, word2):
        vector1 = self.model[word1]
        vector2 = self.model[word2]
        # print(list(vector1))
        # print(list(vector2))
        return self.cosine_similarity(vector1, vector2)

    def get_similar_word_from_text(self, keyword_list: list, text: str, threshold=0.8):
        """
        根据关键词列表从文本中获取相似词汇
        """
        text_split = cutsentences(text).split()
        match_result = []
        for keyword in keyword_list:
            ans = {}
            for index, word in enumerate(text_split):
                try:
                    similarity_score = self.get_word_similarity(keyword, word)
                except Exception as e:
                    logger.info(e)
                    logger.info("模型中没有该词,默认相似度为0")
                    similarity_score = 0
                if similarity_score > threshold:
                    if word not in ans:
                        ans[word] = {
                            "count": 1,
                            "index_position": [index],
                            "similarity_score": similarity_score,
                        }
                    else:
                        ans[word]["count"] += 1
                        ans[word]["index_position"].append(index)
            match_result.append(
                {
                    "keyword": keyword,
                    "similar_result": ans,
                }
            )
        return {"text_split": text_split, "match_result": match_result}


if __name__ == "__main__":
    word2vec_model = (
        "data/word2vec/embedding/tencent-embedding-zh-d100-append-zhihu.bin"
    )
    similarity = WordSimilarity(word2vec_model)
    # result = similarity.get_similar_word_from_text(
    #     ["人工智能", "深度学习"],
    #     "人工智能是深度学习的前提，深度学习是人工智能的一个分支",
    # )
    print(similarity.get_word_similarity("喜欢", "讨厌"))
    # pprint(result)

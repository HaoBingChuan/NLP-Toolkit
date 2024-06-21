#!/usr/bin/python
# -*- coding: UTF-8 -*-

import re
import json
import warnings
import hanlp
import jieba.analyse
from flask import Flask, request, jsonify
from textrank4zh import TextRank4Sentence

from config import (
    ML_MODEL_NAME,
    DL_MODEL_NAME,
    KEYWORD_SIMILAR_WORD2VEC,
    DEVICE,
    BGE_SMALL_MODEL,
)
from data.text_representation import single_sentence_to_vector
from logConfig import logger

from predict import deep_learn_predict, machine_learn_predict
from assis.bizException import BizException
from domain.classificationDomain import ClassificationDomain
from domain.entityRecognitionDomain import EntityRecognitionDomain
from domain.keywordExtractDomain import KeywordExtractDomain
from domain.textSummaryDomain import TextSummaryDomain
from assis.modelManager import ModelManager
from w2ner.predict import w2ner, preload, ParaConfig
from word_similarity import WordSimilarity
from FlagEmbedding import FlagModel
from pycorrector import MacBertCorrector, Corrector

warnings.filterwarnings("ignore")

app = Flask(__name__)

# 预加载分类模型
manager = ModelManager(model_names=["lg", "knn", "svm", "lstm", "cnn"])
# 预加载关键词相似度模型
keyword_similar = WordSimilarity(KEYWORD_SIMILAR_WORD2VEC)
# 预加载hanlp模型
tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH, devices=DEVICE)
ner = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH, devices=DEVICE)
# 预加载TextRank4zh文本摘要提取模型
tr4s = TextRank4Sentence()
# 预加载w2ner实体提取模型
preload_model, tokenizer, vocab = preload(ParaConfig)
# 预加载关键词向量化模型
# Setting use_fp16 to True speeds up computation with a slight performance degradation
bge_small_model = FlagModel(
    BGE_SMALL_MODEL,
    use_fp16=True,
)
# 预加载文本纠错模型
# correct_model = MacBertCorrector("save_model/correct/macbert4csc-base-chinese")
correct_model = Corrector(proper_name_path="save_model/my_custom_proper.txt")


def split_sentences(text):
    # 使用正则表达式匹配句号和问号结尾的句子
    sentences = re.split(r"[。？]", text)
    # 移除分割结果中的空字符串
    sentences = [sentence for sentence in sentences if sentence]
    return sentences


@app.errorhandler(BizException)
def server_exception(ex: BizException):
    return jsonify(get_error(message=ex.message))


def get_error(code=-1, message=""):
    return {"code": code, "message": message}


@app.route("/api/text_category_predict", methods=["POST"])
def text_category_predict():
    """
    开源数据集文本多分类接口
    测试样例：
    {"data":{"texts":["词汇阅读是关键,08年考研暑期英语复习全指南","大学生求爱遭拒杀女同学续受害者母亲网上追忆"],"model":"lstm"}}
    """
    req_data = request.get_data(as_text=True)
    logger.info("text_category_predict 请求数据: param ={} ".format(req_data))

    input_json = ClassificationDomain.dict2entity(json.loads(req_data))

    input_json.check_param()
    texts = input_json.texts
    model = input_json.model
    if model in ML_MODEL_NAME:
        model = manager.find_model(model)
        result = {}
        for text in texts:
            feature = single_sentence_to_vector(text)
            ans = machine_learn_predict([feature], model)
            result[text] = ans[0]
        rs_data = {"data": result, "code": 200, "message": "处理成功"}
        return jsonify(rs_data)
    elif model in DL_MODEL_NAME:
        result = {}
        model = manager.find_model(model)
        ans = deep_learn_predict(texts, model)
        for i in range(len(texts)):
            result[texts[i]] = ans[i]
        rs_data = {"data": result, "code": 200, "message": "处理成功"}
        return jsonify(rs_data)


@app.route("/api/keyword_similarity", methods=["POST"])
def keyword_similarity():
    """
    关键词相似度匹配接口
    测试样例：
    {
    "data":{"keywords":["人工智能","深度学习"],
    "text":"人工智能研究的范围非常广，包括演绎、推理和解决问题、知识表示、学习、运动和控制、数据挖掘等众多领域。其中，知识表示是人工智能领域的核心研究问题之一，它的目标是让机器存储相应的知识，并且能够按照某种规则推理演绎得到新的知识。许多问题的解决都需要先验知识。举个例子，当我们说“林黛玉”，我们会联想到“弱不禁风”“楚楚可怜”的形象，与此同时，我们还会联想到林黛玉的扮演者“陈晓旭”。如果没有先验的知识，我们无法将“林黛玉”和“陈晓旭”关联起来。在这里，“林黛玉”“陈晓旭”都是实体（也称为本体），实体与实体之间通过某种关系连接起来，实体、实体间的关系该如何存储、如何表示、如何方便地应用到生产和生活中，这些都是知识表示要研究的课题"}
    }
    """
    req_data = request.get_data(as_text=True)
    logger.info("keyword_similarity 请求数据: param ={} ".format(req_data))

    input_json = json.loads(req_data)
    if "data" in input_json:
        data_dic = input_json["data"]
        if "text" in data_dic and "keywords" in data_dic:
            texts = data_dic["text"]
            keywords = data_dic["keywords"]
            if texts and keywords:
                result = keyword_similar.get_similar_word_from_text(keywords, texts)
                rs_data = {"data": result, "code": 200, "message": "处理成功"}
                return jsonify(rs_data)
            else:
                return jsonify({"code": -1, "message": "text 不能为空!"})
    return jsonify({"code": -1, "message": "处理异常"})


@app.route("/api/textrank/keyword_extract", methods=["POST"])
def textrank_keyword_extract():
    """
    基于textrank算法进行关键词提取接口
    测试样例：
    {"data":{"text":"瑞华首战复仇心切 中国玫瑰要用美国方式攻克瑞典多曼来了，瑞典来了，商瑞华首战求3分的信心也来了。距离首战72小时当口，中国女足彻底从“恐瑞症”当中获得解脱，因为商瑞华已经找到了瑞典人的软肋。找到软肋，保密4月20日奥运会分组抽签结果出来后，中国姑娘就把瑞典锁定为关乎奥运成败的头号劲敌，因为除了浦玮等个别老将之外，现役女足将士竟然没有人尝过击败瑞典的滋味。在中瑞两队共计15次交锋的历史上，中国队6胜3平6负与瑞典队平分秋色，但从2001年起至今近8年时间，中国在同瑞典连续5次交锋中均未尝胜绩，战绩为2平3负。尽管八年不胜瑞典曾一度成为谢亚龙聘请多曼斯基的理由之一，但这份战绩表也成为压在姑娘们身上的一座大山，在奥运备战过程中越发凸显沉重。或许正源于此，商瑞华才在首战前3天召开了一堂完全针对“恐瑞症”的战术分析课。3日中午在喜来登大酒店中国队租用的会议室里，商瑞华给大家播放了瑞典队的比赛录像剪辑。这是7月6日瑞典在主场同美国队进行的一场奥运热身赛，当时美国队头牌射手瓦姆巴赫还没有受伤，比赛当中双方均尽遣主力，占据主场优势的瑞典队曾一度占据上风，但终究还是在定位球防守上吃了亏，美国队在下半场的一次角球配合中，通过远射打进唯一进球，尽管慢动作显示是打在瑞典队后卫腿上弹射入网，但从过程到结果，均显示了相同内容——“瑞典队的防守并非无懈可击”。商瑞华让科研教练曹晓东等人对这场比赛进行精心剪辑，尤其是瑞典队失球以及美国队形成有威胁射门的片段，更是被放大进行动作分解，每一个中国姑娘都可以一目了然地看清瑞典队哪些地方有机可乘。甚至之前被商瑞华称为“恐怖杀手”的瑞典8号谢琳，也在这次战术分解过程中被发现了不足之处。姑娘们心知肚明并开心享受对手软肋被找到的欢悦，但却必须对记者保密，某主力球员说：“这可不能告诉外界，反正我们心里有数了，知道对付这个速度奇快的谢琳该怎么办！”老帅的“瑞典情结”就像中国队8年不胜瑞典一样，瑞典队也连续遭遇对美国队的溃败：去年阿尔加夫杯瑞典0比1不敌美国，世界杯小组赛上瑞典0比2被美国完胜，今年阿尔加夫杯美国人再次击败瑞典，算上7月6日一役，瑞典遇到美国连平局都没有，竟然是4连败！3日中午的这堂战术分析课后，“用美国人的方式击败瑞典”已经成为中国女足将士的共同心声。姑娘们当然有理由这样去憧憬，因为在7月30日奥运会前最后一场热身赛中，中国女足便曾0比0与强大的美国队握手言和。在3日中午的战术分析课上，这场中美热身也被梳理出片段，与姑娘们一道完成总结。点评过程中，商瑞华对大家在同美国队比赛中表现出来的逼抢意识，给予很高评价。主帅的认可和称赞，更是在大家潜意识里强化了“逼抢的价值”，就连排在18人名单外的候补球员都说：“既然我们能跟最擅长逼抢的美国队玩逼抢，当然有信心跟瑞典队也这么打”，姑娘们都对打好首战的信心越来越强。“我们当然需要低调，但内心深处已经再也没有对瑞典的恐惧，只要把我们训练中的内容打出来，就完全有可能击败瑞典”，这堂战术讨论课后，不止一名球员向记者表达着对首战获胜的渴望。商瑞华心中的“瑞典情结”其实最重，不仅仅因为他是主帅，17年前商瑞华首次担任中国女足主帅时所遭遇的滑铁卢，正是源自1991年世界杯中国0比1被瑞典淘汰所致。抽签出来后面对与瑞典同组，64岁的老帅那份复仇的雄心也潜滋暗长，在5月上旬和7月中旬，商瑞华曾连续两次前往欧洲在现场刺探瑞典军情，真正对瑞典队的特点达到了如指掌的程度。昨天中午的战术讨论课后，商瑞华告诉记者：“瑞典队和所有其他队伍一样有优点也有不足，我们如果能够扬长避短拿对方的不足做文章，就有可能击败对手。从奥运会备战阶段看来，中国队战术上打得比较快的特点基本已经成型，边路也不错，在前场抢截后快速发动进攻的能力也逐步增强。我很清楚瑞典队世界排名第3而我们排第14，但承认差距不等于接受失败，我当然想赢。”"}}
    """
    req_data = request.get_data(as_text=True)
    logger.info("keyword_extract 请求数据: param ={} ".format(req_data))

    input_json = KeywordExtractDomain.dict2entity(json.loads(req_data))
    text = input_json.text
    topK = input_json.topK
    allowPOS = input_json.allowPOS
    result = jieba.analyse.textrank(text, topK=topK, withWeight=True, allowPOS=allowPOS)
    rs_data = {
        "data": [w for w, v in result],
        "code": 200,
        "message": "处理成功",
    }
    return jsonify(rs_data)


@app.route("/api/ner", methods=["POST"])
def ner_predict():
    """
    实体识别,基于Hanlp/w2ner进行实体识别任务
    测试样例：
    {"data":{"text":"瑞华首战复仇心切 中国玫瑰要用美国方式攻克瑞典多曼来了，瑞典来了，商瑞华首战求3分的信心也来了。距离首战72小时当口，中国女足彻底从“恐瑞症”当中获得解脱，因为商瑞华已经找到了瑞典人的软肋。找到软肋，保密4月20日奥运会分组抽签结果出来后，中国姑娘就把瑞典锁定为关乎奥运成败的头号劲敌，因为除了浦玮等个别老将之外，现役女足将士竟然没有人尝过击败瑞典的滋味。在中瑞两队共计15次交锋的历史上，中国队6胜3平6负与瑞典队平分秋色，但从2001年起至今近8年时间，中国在同瑞典连续5次交锋中均未尝胜绩，战绩为2平3负。尽管八年不胜瑞典曾一度成为谢亚龙聘请多曼斯基的理由之一，但这份战绩表也成为压在姑娘们身上的一座大山，在奥运备战过程中越发凸显沉重。或许正源于此，商瑞华才在首战前3天召开了一堂完全针对“恐瑞症”的战术分析课。3日中午在喜来登大酒店中国队租用的会议室里，商瑞华给大家播放了瑞典队的比赛录像剪辑。这是7月6日瑞典在主场同美国队进行的一场奥运热身赛，当时美国队头牌射手瓦姆巴赫还没有受伤，比赛当中双方均尽遣主力，占据主场优势的瑞典队曾一度占据上风，但终究还是在定位球防守上吃了亏，美国队在下半场的一次角球配合中，通过远射打进唯一进球，尽管慢动作显示是打在瑞典队后卫腿上弹射入网，但从过程到结果，均显示了相同内容——“瑞典队的防守并非无懈可击”。商瑞华让科研教练曹晓东等人对这场比赛进行精心剪辑，尤其是瑞典队失球以及美国队形成有威胁射门的片段，更是被放大进行动作分解，每一个中国姑娘都可以一目了然地看清瑞典队哪些地方有机可乘。甚至之前被商瑞华称为“恐怖杀手”的瑞典8号谢琳，也在这次战术分解过程中被发现了不足之处。姑娘们心知肚明并开心享受对手软肋被找到的欢悦，但却必须对记者保密，某主力球员说：“这可不能告诉外界，反正我们心里有数了，知道对付这个速度奇快的谢琳该怎么办！”老帅的“瑞典情结”就像中国队8年不胜瑞典一样，瑞典队也连续遭遇对美国队的溃败：去年阿尔加夫杯瑞典0比1不敌美国，世界杯小组赛上瑞典0比2被美国完胜，今年阿尔加夫杯美国人再次击败瑞典，算上7月6日一役，瑞典遇到美国连平局都没有，竟然是4连败！3日中午的这堂战术分析课后，“用美国人的方式击败瑞典”已经成为中国女足将士的共同心声。姑娘们当然有理由这样去憧憬，因为在7月30日奥运会前最后一场热身赛中，中国女足便曾0比0与强大的美国队握手言和。在3日中午的战术分析课上，这场中美热身也被梳理出片段，与姑娘们一道完成总结。点评过程中，商瑞华对大家在同美国队比赛中表现出来的逼抢意识，给予很高评价。主帅的认可和称赞，更是在大家潜意识里强化了“逼抢的价值”，就连排在18人名单外的候补球员都说：“既然我们能跟最擅长逼抢的美国队玩逼抢，当然有信心跟瑞典队也这么打”，姑娘们都对打好首战的信心越来越强。“我们当然需要低调，但内心深处已经再也没有对瑞典的恐惧，只要把我们训练中的内容打出来，就完全有可能击败瑞典”，这堂战术讨论课后，不止一名球员向记者表达着对首战获胜的渴望。商瑞华心中的“瑞典情结”其实最重，不仅仅因为他是主帅，17年前商瑞华首次担任中国女足主帅时所遭遇的滑铁卢，正是源自1991年世界杯中国0比1被瑞典淘汰所致。抽签出来后面对与瑞典同组，64岁的老帅那份复仇的雄心也潜滋暗长，在5月上旬和7月中旬，商瑞华曾连续两次前往欧洲在现场刺探瑞典军情，真正对瑞典队的特点达到了如指掌的程度。昨天中午的战术讨论课后，商瑞华告诉记者：“瑞典队和所有其他队伍一样有优点也有不足，我们如果能够扬长避短拿对方的不足做文章，就有可能击败对手。从奥运会备战阶段看来，中国队战术上打得比较快的特点基本已经成型，边路也不错，在前场抢截后快速发动进攻的能力也逐步增强。我很清楚瑞典队世界排名第3而我们排第14，但承认差距不等于接受失败，我当然想赢。","entity_type":["PERSON", "ORGANIZATION"]}}
    """
    req_data = request.get_data(as_text=True)
    logger.info("ner 请求数据: param ={} ".format(req_data))

    input_json = EntityRecognitionDomain.dict2entity(json.loads(req_data))
    input_json.check_param()
    model_type = input_json.model_type
    entity_type = input_json.entity_type
    text = input_json.text
    text_sentences = split_sentences(text)
    ans = {}
    if model_type == "hanlp":
        for sentence in text_sentences:
            text_split = tok(sentence)
            for w in ner(text_split):
                if w[1] in entity_type:
                    word = w[0]
                    entity = w[1]
                    if word not in ans:
                        ans[word] = {"entity": entity, "count": 1}
                    else:
                        ans[word]["count"] += 1
    elif model_type == "w2ner":
        for sentence in text_sentences:
            info = w2ner(sentence, preload_model, tokenizer, vocab)
            for e, t in zip(info["entities"], info["entity_types"]):
                if t in entity_type:
                    word = e
                    entity = t
                    if word not in ans:
                        ans[word] = {"entity": entity, "count": 1}
                    else:
                        ans[word]["count"] += 1
    result = [
        {"word": word, "entity": v["entity"], "count": v["count"]}
        for word, v in ans.items()
    ]
    rs_data = {
        "data": result,
        "code": 200,
        "message": "处理成功",
    }
    return jsonify(rs_data)


@app.route("/api/text_summary", methods=["POST"])
def text_summary():
    """
    文本摘要提取,基于TextRank4zh进行文本摘要提取任务
    测试样例：
    {"data":{"text":"瑞华首战复仇心切 中国玫瑰要用美国方式攻克瑞典多曼来了，瑞典来了，商瑞华首战求3分的信心也来了。距离首战72小时当口，中国女足彻底从“恐瑞症”当中获得解脱，因为商瑞华已经找到了瑞典人的软肋。找到软肋，保密4月20日奥运会分组抽签结果出来后，中国姑娘就把瑞典锁定为关乎奥运成败的头号劲敌，因为除了浦玮等个别老将之外，现役女足将士竟然没有人尝过击败瑞典的滋味。在中瑞两队共计15次交锋的历史上，中国队6胜3平6负与瑞典队平分秋色，但从2001年起至今近8年时间，中国在同瑞典连续5次交锋中均未尝胜绩，战绩为2平3负。尽管八年不胜瑞典曾一度成为谢亚龙聘请多曼斯基的理由之一，但这份战绩表也成为压在姑娘们身上的一座大山，在奥运备战过程中越发凸显沉重。或许正源于此，商瑞华才在首战前3天召开了一堂完全针对“恐瑞症”的战术分析课。3日中午在喜来登大酒店中国队租用的会议室里，商瑞华给大家播放了瑞典队的比赛录像剪辑。这是7月6日瑞典在主场同美国队进行的一场奥运热身赛，当时美国队头牌射手瓦姆巴赫还没有受伤，比赛当中双方均尽遣主力，占据主场优势的瑞典队曾一度占据上风，但终究还是在定位球防守上吃了亏，美国队在下半场的一次角球配合中，通过远射打进唯一进球，尽管慢动作显示是打在瑞典队后卫腿上弹射入网，但从过程到结果，均显示了相同内容——“瑞典队的防守并非无懈可击”。商瑞华让科研教练曹晓东等人对这场比赛进行精心剪辑，尤其是瑞典队失球以及美国队形成有威胁射门的片段，更是被放大进行动作分解，每一个中国姑娘都可以一目了然地看清瑞典队哪些地方有机可乘。甚至之前被商瑞华称为“恐怖杀手”的瑞典8号谢琳，也在这次战术分解过程中被发现了不足之处。姑娘们心知肚明并开心享受对手软肋被找到的欢悦，但却必须对记者保密，某主力球员说：“这可不能告诉外界，反正我们心里有数了，知道对付这个速度奇快的谢琳该怎么办！”老帅的“瑞典情结”就像中国队8年不胜瑞典一样，瑞典队也连续遭遇对美国队的溃败：去年阿尔加夫杯瑞典0比1不敌美国，世界杯小组赛上瑞典0比2被美国完胜，今年阿尔加夫杯美国人再次击败瑞典，算上7月6日一役，瑞典遇到美国连平局都没有，竟然是4连败！3日中午的这堂战术分析课后，“用美国人的方式击败瑞典”已经成为中国女足将士的共同心声。姑娘们当然有理由这样去憧憬，因为在7月30日奥运会前最后一场热身赛中，中国女足便曾0比0与强大的美国队握手言和。在3日中午的战术分析课上，这场中美热身也被梳理出片段，与姑娘们一道完成总结。点评过程中，商瑞华对大家在同美国队比赛中表现出来的逼抢意识，给予很高评价。主帅的认可和称赞，更是在大家潜意识里强化了“逼抢的价值”，就连排在18人名单外的候补球员都说：“既然我们能跟最擅长逼抢的美国队玩逼抢，当然有信心跟瑞典队也这么打”，姑娘们都对打好首战的信心越来越强。“我们当然需要低调，但内心深处已经再也没有对瑞典的恐惧，只要把我们训练中的内容打出来，就完全有可能击败瑞典”，这堂战术讨论课后，不止一名球员向记者表达着对首战获胜的渴望。商瑞华心中的“瑞典情结”其实最重，不仅仅因为他是主帅，17年前商瑞华首次担任中国女足主帅时所遭遇的滑铁卢，正是源自1991年世界杯中国0比1被瑞典淘汰所致。抽签出来后面对与瑞典同组，64岁的老帅那份复仇的雄心也潜滋暗长，在5月上旬和7月中旬，商瑞华曾连续两次前往欧洲在现场刺探瑞典军情，真正对瑞典队的特点达到了如指掌的程度。昨天中午的战术讨论课后，商瑞华告诉记者：“瑞典队和所有其他队伍一样有优点也有不足，我们如果能够扬长避短拿对方的不足做文章，就有可能击败对手。从奥运会备战阶段看来，中国队战术上打得比较快的特点基本已经成型，边路也不错，在前场抢截后快速发动进攻的能力也逐步增强。我很清楚瑞典队世界排名第3而我们排第14，但承认差距不等于接受失败，我当然想赢。"}}
    """
    req_data = request.get_data(as_text=True)
    logger.info("text_summary 请求数据: param ={} ".format(req_data))

    input_json = TextSummaryDomain.dict2entity(json.loads(req_data))
    input_json.check_param()

    text = input_json.text
    num = input_json.num
    tr4s.analyze(text=text)
    rs_data = {
        "data": [item.sentence for item in tr4s.get_key_sentences(num=num)],
        "code": 200,
        "message": "处理成功",
    }
    return jsonify(rs_data)


@app.route("/api/text_vector", methods=["POST"])
def text_vector():
    """
    文本向量化
    :return:
    """
    try:
        req_data = request.get_data(as_text=True)
        logger.info("text_vector 请求数据: param ={} ".format(req_data))
        req_data = json.loads(req_data)
        keyword_list = req_data["data"]["keywords"]
        vector_info_list = []
        for keyword in keyword_list:
            vector = bge_small_model.encode(keyword)
            vector = vector.tolist()
            vector_info = {"text": keyword, "vector": vector}
            vector_info_list.append(vector_info)
        rs_data = {
            "data": vector_info_list,
            "code": 200,
            "message": "处理成功",
        }
        return jsonify(rs_data)
    except Exception as ex:
        logger.error(ex)
        rs_data = {
            "code": 500,
            "message": ex,
        }
        return jsonify(rs_data)


@app.route("/api/text_correct", methods=["POST"])
def text_correct():
    """
    文本纠错接口
    测试样例：
    {"text":"我从北京南做高铁到南京南"}
    """
    req_data = request.get_data(as_text=True)
    logger.info("text_correct 请求数据: param ={} ".format(req_data))
    input_json = json.loads(req_data)
    text = input_json["text"]
    info = correct_model.correct(text)
    errors = []
    for i, item in enumerate(info["source"]):
        if item != info["target"][i]:
            error = {"position": i, "correction": {item: info["target"][i]}}
            errors.append(error)
    rs_data = {"source": info["source"], "target": info["target"], "errors": errors}
    return jsonify(rs_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8070)

# 本项目主要包含三部分功能：
  1.词向量训练 
  2.关键词相似度匹配 
  3.机器学习文本分类 以及多个RESTful接口(文本分类、关键词相似度匹配、关键词提取、实体识别、文本摘要提取)

# 词向量训练
    训练代码：word2vec_train.py
    训练数据：data/word2vec/
    功能：1.基于文本训练词向量 2.基于已训练完成的词向量增量训练 3.基于开源词向量进行增量训练

# 关键词相似度匹配
    代码：word_similarity.py
    功能：1.基于词向量计算两个词的相似度 2.根据关键词列表从文本中获取相似词汇

# 机器学习文本分类


- [本项目主要包含三部分功能：](#本项目主要包含三部分功能)
- [词向量训练](#词向量训练)
- [关键词相似度匹配](#关键词相似度匹配)
- [机器学习文本分类](#机器学习文本分类)
  - [关于 ](#关于-)
    - [1. 介绍](#1-介绍)
    - [2. 目前已经涵盖的算法](#2-目前已经涵盖的算法)
      - [2.1 常见的机器学习算法](#21-常见的机器学习算法)
      - [2.2 常见的深度学习算法](#22-常见的深度学习算法)
  - [前提准备 ](#前提准备-)
    - [环境安装](#环境安装)
  - [具体使用方法 ](#具体使用方法-)
    - [3. 参数介绍](#3-参数介绍)
    - [3.1 针对常见的机器学习算法](#31-针对常见的机器学习算法)
    - [3.2 针对深度神经网络算法](#32-针对深度神经网络算法)
    - [Note：](#note)


---

## 关于 <a name = "关于"></a>
### 1. 介绍

> 这是一个包含多种机器学习算法的代码，其主要用于NLP中文本分类的下游任务，包括二分类及多分类。使用者需更改一些参数例如数据集地址，算法名称等，即可以使用其中的各种模型来进行文本分类，各种算法的参数只在xx_config.py单个文件中提供，方便对神经网络模型进行调参。
### 2. 目前已经涵盖的算法
#### 2.1 常见的机器学习算法

- Logistic Regression
- KNN
- Decision Tree
- Random Forest
- GBDT(Gradient Boosting Decision Tree)
- XGBoost
- Catboost
- SVM
- Bayes
- todo...


#### 2.2 常见的深度学习算法

- TextCNN
- Bi-LSTM
- Transformer
- Capsules
- todo...





## 前提准备 <a name = "前提准备"></a>

### 环境安装

具体的相关库的版本见requestments.txt

- 使用命令安装

```
pip install -r requestments.txt
```



## 具体使用方法 <a name = "具体使用方法"></a>
<br>

### 3. 参数介绍
***主程序：main.py，其中各个参数的含义如下：***


> *--model_name*: 需要使用的算法名称，填写的简称见config.py中的ML_MODEL_NAME和DL_MODEL_NAME
> 
> *--model_saved_path*: 模型存储的路径
> 
> *--type_obj*: 程序的运行目的：train，test，predict三个选项
> 
> *--train_data_path*: 切分好的训练集路径
>
> *--test_data_path*: 切分好的测试集路径
> 
> *--dev_data_path*: 切分好的验证集路径
### 3.1 针对常见的机器学习算法


***多分类示例(机器学习建议数据量不宜过大，否则加载困难)***

```
# 训练及测试(更改--test_data_path参数实现不同文件的测试)
python main.py --train_data_path ./data/ml_data/test_feature_word2vec.csv --test_data_path ./data/ml_data/dev_feature_word2vec.csv --model_saved_path ./save_model/ --model_name lg --type_obj train
```


### 3.2 针对深度神经网络算法


***示例***

```
# 训练代码
# python main.py --model_name cnn --model_saved_path ./save_model/ --type_obj train --train_data_path ./data/dl_data/train.csv --test_data_path ./data/dl_data/test.csv
# 测试代码
# python main.py --model_name cnn --model_saved_path ./save_model/ --type_obj test --test_data_path ./data/dl_data/test.csv

```

### Note：
>> **常见的机器学习算法调参在 ml_algorithm/ml_moel.py下<br>深度神经网络/预训练模型的调参在 dl_algorithm/dl_config.py下<br>其他全局参数调参在 ./config.py下


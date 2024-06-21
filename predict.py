import os
import sys

sys.path.append(os.getcwd())
from process_data.deepLearnDataExecutor import DeepLearnDataExecutor
from config import ML_MODEL_NAME, DL_MODEL_NAME, BATCH_SIZE, SPLIT_SIZE, IS_SAMPLE
from assis.modelManager import ModelManager


def deep_learn_predict(text_data, model_ex):
    data_ex = DeepLearnDataExecutor()
    data_ex.process(batch_size=BATCH_SIZE, dev_data_path=text_data)
    y_pre_test = model_ex.predict(data_ex.dev_data_loader)
    i2l_dict = data_ex.i2l_dic  # 可以将y_pre_test中的数字转成文字标签，按需使用
    return [i2l_dict[i] for i in y_pre_test]


def machine_learn_predict(text_data, model_ex):
    y_pre_test = model_ex.predict(text_data)
    return y_pre_test


if __name__ == "__main__":
    # 测试代码
    text_data = [
        "词汇阅读是关键,08年考研暑期英语复习全指南",
        "大学生求爱遭拒杀女同学续受害者母亲网上追忆",
    ]
    model_ex = ModelManager(model_names=["lg"]).find_model("lg")
    print(deep_learn_predict(text_data, model_ex=model_ex))

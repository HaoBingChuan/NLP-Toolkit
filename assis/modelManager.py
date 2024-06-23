import os
import sys
sys.path.append(os.getcwd())
from process_data.deepLearnDataExecutor import DeepLearnDataExecutor
from dl_algorithm.dl_config import DlConfig
from model import ModelExecutor
from config import ML_MODEL_NAME, DL_MODEL_NAME, BATCH_SIZE
import joblib


class ModelManager(object):
    dic_model = {}
    model_saved_path = "./save_model"

    def __init__(self, model_names: list):
        for item in model_names:
            if item in DL_MODEL_NAME:
                self.deep_learn_reload(item)
            elif item in ML_MODEL_NAME:
                self.machine_learn_reload(item)
            else:
                raise Exception(f"{item} 模型不存在")

    def deep_learn_reload(self, model_name: str):
        data_ex = DeepLearnDataExecutor()
        vocab_size, nums_class = data_ex.process(batch_size=BATCH_SIZE)
        dl_config = DlConfig(model_name, vocab_size, nums_class, data_ex.vocab)
        # 初始化模型
        model_ex = ModelExecutor().init(dl_config=dl_config)
        model_ex.load_model(self.model_saved_path, model_name + ".pth")

        self.dic_model[model_name] = model_ex
        # logger.info(f"{model_name} 模型加载完成")

    def machine_learn_reload(self, model_name: str):
        model = joblib.load(f"{self.model_saved_path}/{model_name}.pkl")
        self.dic_model[model_name] = model

    def find_model(self, code: str):
        if code in self.dic_model:
            return self.dic_model[code]
        return None

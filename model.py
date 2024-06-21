#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from config import ML_MODEL_NAME, DL_MODEL_NAME
from ml_algorithm.machineLearnExecutor import MachineLearnExecutor
from dl_algorithm.deepLearnExecutor import DeepLearnExecutor


class ModelExecutor:
    def __init__(self):
        pass

    def init(self, model_name="", dl_config=""):
        if model_name in ML_MODEL_NAME:
            return MachineLearnExecutor(model_name)
        elif dl_config.model_name in DL_MODEL_NAME:
            return DeepLearnExecutor(dl_config)

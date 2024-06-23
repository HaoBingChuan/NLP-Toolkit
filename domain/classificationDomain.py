from assis.bizException import BizException
from config import ML_MODEL_NAME, DL_MODEL_NAME


class ClassificationDomain(object):
    """
    classification实体对象
    """

    def __init__(
        self,
        data: str = None,
        texts: list = [],
        model: str = "cnn",
    ):
        self.data = data
        self.texts = texts
        self.model = model

    @staticmethod
    def dict2entity(dic):
        instance = ClassificationDomain()
        if "data" in dic:
            instance.data = dic["data"]
            if "texts" in dic["data"]:
                instance.texts = dic["data"]["texts"]
            if "model" in dic["data"]:
                instance.model = dic["data"]["model"]
        return instance

    def check_param(self):
        if not self.data:
            raise BizException("data 不能为空!")
        if not self.texts:
            raise BizException("texts 不能为空!")
        if self.model not in ML_MODEL_NAME and self.model not in DL_MODEL_NAME:
            raise BizException("model_type 参数错误!")

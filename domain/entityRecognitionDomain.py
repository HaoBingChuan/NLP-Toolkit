from assis.bizException import BizException


class EntityRecognitionDomain(object):
    """
    ner实体对象
    entity_type默认识别类型：["PERSON", "ORGANIZATION"]；支持类型：["PERSON", "ORGANIZATION","LOCATION","DATE"]
    """

    def __init__(
        self,
        data: str = None,
        text: str = None,
        entity_type: list = ["PERSON", "ORGANIZATION"],
        model_type: str = "w2ner",
    ):
        self.data = data
        self.text = text
        self.entity_type = entity_type
        self.model_type = model_type

    @staticmethod
    def dict2entity(dic):
        instance = EntityRecognitionDomain()
        if "data" in dic:
            instance.data = dic["data"]
            if "text" in dic["data"]:
                instance.text = dic["data"]["text"]
            if "entity_type" in dic["data"]:
                instance.entity_type = dic["data"]["entity_type"]
            if "model_type" in dic["data"]:
                instance.model_type = dic["data"]["model_type"]
        return instance

    def check_param(self):
        if not self.data:
            raise BizException("data 不能为空!")
        if not self.text:
            raise BizException("text 不能为空!")
        if not self.entity_type:
            raise BizException("entity_type 不能为空!")
        if not self.model_type:
            raise BizException("model_type 不能为空!")

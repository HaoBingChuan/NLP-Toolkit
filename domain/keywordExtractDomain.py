from assis.bizException import BizException


class KeywordExtractDomain(object):
    """
    关键词提取实体对象
    词性默认识别类型：["ns", "n", "vn", "v"]；可选支持类型：["ns", "n", "vn", "v","m","q","nz"]
    """

    def __init__(
        self,
        data: str = None,
        text: str = None,
        topK: int = 20,
        allowPOS: set = ("ns", "n", "vn", "v"),
    ):
        self.data = data
        self.text = text
        self.topK = topK
        self.allowPOS = allowPOS

    @staticmethod
    def dict2entity(dic):
        instance = KeywordExtractDomain()
        if "data" in dic:
            instance.data = dic["data"]
            if "text" in dic["data"]:
                instance.text = dic["data"]["text"]
            if "topK" in dic["data"]:
                instance.topK = dic["data"]["topK"]
            if "allowPOS" in dic["data"]:
                instance.allowPOS = dic["data"]["allowPOS"]
        return instance

    def check_param(self):
        if not self.data:
            raise BizException("data 不能为空!")
        if not self.text:
            raise BizException("text 不能为空!")
        if not self.topK:
            raise BizException("topK 必须大于0")
        if not self.allowPOS:
            raise BizException("allowPOS 不能为空!")

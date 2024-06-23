from assis.bizException import BizException


class TextSummaryDomain(object):
    """
    文本摘要提取实体对象
    默认提取句子数量num = 6
    """

    def __init__(
        self,
        data: str = None,
        text: str = None,
        num: int = 6,
    ):
        self.data = data
        self.text = text
        self.num = num

    @staticmethod
    def dict2entity(dic):
        instance = TextSummaryDomain()
        if "data" in dic:
            instance.data = dic["data"]
            if "text" in dic["data"]:
                instance.text = dic["data"]["text"]
            if "num" in dic["data"]:
                instance.num = dic["data"]["num"]
        return instance

    def check_param(self):
        if not self.data:
            raise BizException("data 不能为空!")
        if not self.text:
            raise BizException("text 不能为空!")
        if not self.num:
            raise BizException("num 必须大于0")

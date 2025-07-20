class Qwen25VL72BInstruct:
    def __init__(self):
        self.model = "Qwen/Qwen2.5-VL-72B-Instruct"
        self.api_key = "your modelscope api"
        self.api_base = "https://api-inference.modelscope.cn/v1/"

class Qwen3_235B_A22B:
    def __init__(self):
        self.model = "Qwen/Qwen3-235B-A22B"
        self.api_key = "your modelscope api"
        self.api_base = "https://api-inference.modelscope.cn/v1/"

class GPT41:
    def __init__(self):
        self.model = "gpt-4.1"
        self.api_key = "your dmxapi api"
        self.api_base = "https://www.dmxapi.cn/v1"

class Bge_Reranker_v2_m3:
    def __init__(self) -> None:
        self.model = "BAAI/bge-reranker-v2-m3"
        self.api_key = "your siliconflow api"
        self.api_base = "https://api.siliconflow.cn/v1"

class AliBailian_API:
    def __init__(self) -> None:
        self.api_key = "your alibailian api"

class Cohere_API:
    def __init__(self) -> None:
        self.api_key = "your cohere api"

class Google_API:
    def __init__(self) -> None:
        self.api_key = "your google ai studio api"
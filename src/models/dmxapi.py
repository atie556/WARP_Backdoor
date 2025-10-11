import os
from openai import OpenAI
from .Model import Model

class dmxapi(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        self.max_output_tokens = config["params"]["max_output_tokens"]

        api_pos = int(config["api_key_info"]["api_key_use"])
        self.api_key = config["api_key_info"]["api_keys"][api_pos]

        # 使用 OpenAI API 来初始化模型和 tokenizer
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deerapi.com/v1")
        self.model_name = config["model_info"]["name"]
# https://www.dmxapi.com/v1   https://api.deerapi.com/v1
    #def query(self, msg):
    #    # 使用 OpenAI API 创建聊天请求并获取响应
    #    response = self.client.chat.completions.create(
    #        model=self.model_name,
    #        messages=[{'role': 'user', 'content': msg}],
    #        temperature=self.temperature,
    #        max_tokens=self.max_output_tokens,
    #        stream=True
    #    )
#
    #    # 处理流式响应
    #    result = ""
    #    for chunk in response:
    #        # 安全获取各层属性（兼容性写法）
    #        choices = getattr(chunk, "choices", [{}])  # 默认给一个空字典避免索引错误
    #        first_choice = choices[0] if len(choices) > 0 else {}
    #        delta = getattr(first_choice, "delta", {})
    #        content = getattr(delta, "content", "") or ""  # 处理content为None的情况
#
    #        result += content
#
    #    return result
    def query(self, msg):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{'role': 'user', 'content': msg}],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            stream=False
        )

        # ✅ 安全提取内容
        if hasattr(response, "output") and isinstance(response.output, list) and len(response.output) > 0:
            content = response.output[0].get("content", "")
        elif hasattr(response, "choices") and len(response.choices) > 0:
            # 兼容旧版 OpenAI 风格
            content = getattr(response.choices[0].message, "content", "")
        else:
            content = str(response)

        return content.strip() if content else "[EMPTY OUTPUT]"


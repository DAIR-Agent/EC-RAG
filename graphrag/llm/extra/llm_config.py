# -*- encoding: utf-8 -*-
"""
@author: acedar  
@time: 2024/7/27 17:59
@file: llm_config.py 
"""

import os
import abc


class LlmConfig(metaclass=abc.ABCMeta):
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME")
        self.embed_model_name = os.getenv("EMBEDDINGS_MODEL_NAME")

    @abc.abstractmethod
    def get_llm_options(self):
        pass

    @abc.abstractmethod
    def get_embed_options(self):
        pass


class QianFanLlmConfig(LlmConfig):
    # https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Nlks5zkzu

    def __init__(self):
        super(QianFanLlmConfig, self).__init__()
        self.qianfan_ak = os.getenv('QIANFAN_AK')
        self.qianfan_sk = os.getenv('QIANFAN_SK')

    def get_llm_options(self):
        return {
            # 'model': self.model_name,
            'qianfan_ak': self.qianfan_ak,
            'qianfan_sk': self.qianfan_sk
        }

    def get_embed_options(self):
        return {
            # 'model': self.embed_model_name,
            'qianfan_ak': self.qianfan_ak,
            'qianfan_sk': self.qianfan_sk
        }


class TongYiLlmConfig(LlmConfig):
    # https://help.aliyun.com/zh/model-studio/user-guide/tongyi-qianwen
    def __init__(self):
        super(TongYiLlmConfig, self).__init__()
        self.tong_yi_api_key = os.getenv("TONGYI_API_KEY")

    def get_llm_options(self):
        return {
            # 'model': self.model_name,
            'dashscope_api_key': self.tong_yi_api_key
        }

    def get_embed_options(self):
        return {
            # 'model': self.embed_model_name,
            'dashscope_api_key': self.tong_yi_api_key
        }


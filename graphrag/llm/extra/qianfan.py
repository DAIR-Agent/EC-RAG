#!/usr/bin/env python3
# coding=utf-8

"""
Baidu Qianfan: https://qianfan.cloud.baidu.com/
"""

import os
from typing import Union

from langchain_community.chat_models import QianfanChatEndpoint
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.llms import QianfanLLMEndpoint
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel, BaseChatModel
from langchain_core.messages import BaseMessage
from .llm_config import QianFanLlmConfig

# INSTRUCT_MODEL = 'ERNIE-Speed-8K'
# # INSTRUCT_MODEL = 'ERNIE-4.0-Turbo-8K'
# CHAT_MODEL = INSTRUCT_MODEL
# EMBEDDINGS_MODEL = 'bge-large-zh'
#
# common_options = {
#     'qianfan_ak': os.getenv('QIANFAN_AK'),
#     'qianfan_sk': os.getenv('QIANFAN_SK')
# }
#
# _llm, _chat_llm, _embeddings = None, None, None


def create_llm(**kwargs) -> BaseLanguageModel[Union[str, BaseMessage]]:
    """create `QianfanLLM`, can be used to replace `OpenAI`"""
    global _llm, _llm_config
    if _llm:
        return _llm
    _llm_config = _llm_config if _llm_config else QianFanLlmConfig()
    common_options = _llm_config.get_llm_options()
    options = {**common_options, **kwargs}
    print("create_llm:", options)
    _llm = QianfanLLMEndpoint(**options)
    return _llm


def create_chat_llm(**kwargs) -> BaseChatModel:
    """create `QianfanChat`, can be used to replace `ChatOpenAI`"""
    global _chat_llm, _llm_config
    if _chat_llm:
        return _chat_llm
    _llm_config = _llm_config if _llm_config else QianFanLlmConfig()
    common_options = _llm_config.get_llm_options()
    options = {**common_options, **kwargs}
    print("create_llm:", options)
    _chat_llm = QianfanChatEndpoint(**options)
    return _chat_llm


def create_embeddings(**kwargs) -> Embeddings:
    """create `QianfanEmbeddings`, can be used to replace `OpenAIEmbeddings`"""
    global _embeddings
    global _embeddings, _llm_config
    if _embeddings:
        return _embeddings
    _llm_config = _llm_config if _llm_config else QianFanLlmConfig()
    common_options = _llm_config.get_llm_options()
    options = {**common_options, **kwargs}
    print("create_llm:", options)
    _embeddings = QianfanEmbeddingsEndpoint(**options)
    return _embeddings


creators = (create_llm, create_chat_llm, create_embeddings)

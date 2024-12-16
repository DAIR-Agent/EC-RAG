#!/usr/bin/env python3
# coding=utf-8

"""
Alibaba Tongyi: https://tongyi.aliyun.com/
"""

import os
from typing import Union

from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel, BaseChatModel
from langchain_core.messages import BaseMessage
from .llm_config import TongYiLlmConfig
# https://help.aliyun.com/zh/model-studio/user-guide/tongyi-qianwen
# INSTRUCT_MODEL = 'qwen-turbo'
# CHAT_MODEL = INSTRUCT_MODEL
# EMBEDDINGS_MODEL = 'text-embedding-v2'

# os.environ["TONGYI_API_KEY"] = "sk-nadqD6Zbsv"

# common_options = {
#     'dashscope_api_key': os.getenv('TONGYI_API_KEY')
# }

_llm, _chat_llm, _embeddings = None, None, None
_llm_config, _embed_config = None, None


def create_llm(**kwargs) -> BaseLanguageModel[Union[str, BaseMessage]]:
    """create `Tongyi`, can be used to replace `OpenAI`"""
    global _llm, _llm_config
    if _llm:
        return _llm
    _llm_config = _llm_config if _llm_config else TongYiLlmConfig()
    common_options = _llm_config.get_llm_options()
    options = {**common_options, **kwargs}
    print("create_llm:", options)
    _llm = Tongyi(**options)
    return _llm


def create_chat_llm(**kwargs) -> BaseChatModel:
    """create `ChatTongyi`, can be used to replace `ChatOpenAI`"""
    global _chat_llm, _llm_config
    if _chat_llm:
        return _chat_llm
    _llm_config = _llm_config if _llm_config else TongYiLlmConfig()
    common_options = _llm_config.get_llm_options()
    options = {**common_options, **kwargs}
    print("create_chat_llm:", options)
    _chat_llm = ChatTongyi(**options)
    return _chat_llm


def create_embeddings(**kwargs) -> Embeddings:
    """create `DashScopeEmbeddings`, can be used to replace `OpenAIEmbeddings`"""
    global _embeddings, _llm_config
    if _embeddings:
        return _embeddings

    _llm_config = _llm_config if _llm_config else TongYiLlmConfig()
    common_options = _llm_config.get_embed_options()
    options = {**common_options, **kwargs}
    print("create_embeddings:", options)
    _embeddings = DashScopeEmbeddings(**options)
    return _embeddings


creators = (create_llm, create_chat_llm, create_embeddings)

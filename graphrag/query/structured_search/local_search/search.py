# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""LocalSearch implementation."""

import logging
import time
from typing import Any
import pandas as pd

import tiktoken

from graphrag.query.context_builder.builders import LocalContextBuilder
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
)
from graphrag.query.llm.base import BaseLLM, BaseLLMCallback
from graphrag.query.llm.text_utils import num_tokens
from graphrag.query.structured_search.base import BaseSearch, SearchResult
from graphrag.query.structured_search.local_search.system_prompt import (
    LOCAL_SEARCH_SYSTEM_PROMPT,
)

from graphrag.query.structured_search.local_search.refine_prompt import (
    REFINE_PROMPT,
)

import re
 
def remove_duplicate_sentences(text):
    # 使用正则表达式来匹配句子
    sentence_regex = r'[^!?。!?。]+[!?。!?。]'
    sentences = re.findall(sentence_regex, text)
    unique_sentences = list(set(sentences))
    return ' '.join(unique_sentences)


DEFAULT_LLM_PARAMS = {
    "max_tokens": 1500,
    "temperature": 0.0,
}

log = logging.getLogger(__name__)


class LocalSearch(BaseSearch):
    """Search orchestration for local search mode."""

    def __init__(
        self,
        llm: BaseLLM,
        context_builder: LocalContextBuilder,
        token_encoder: tiktoken.Encoding | None = None,
        system_prompt: str = LOCAL_SEARCH_SYSTEM_PROMPT,
        response_type: str = "multiple paragraphs",
        callbacks: list[BaseLLMCallback] | None = None,
        llm_params: dict[str, Any] = DEFAULT_LLM_PARAMS,
        context_builder_params: dict | None = None,
    ):
        super().__init__(
            llm=llm,
            context_builder=context_builder,
            token_encoder=token_encoder,
            llm_params=llm_params,
            context_builder_params=context_builder_params or {},
        )
        self.system_prompt = system_prompt
        self.callbacks = callbacks
        self.response_type = response_type

    async def asearch(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> SearchResult:
        """Build local search context that fits a single context window and generate answer for the user query."""
        start_time = time.time()
        search_prompt = ""

        if conversation_history:
            query_graphrag = self.refine_with_conversation_history(query, conversation_history)
        else:
            query_graphrag = query

        context_text, context_records = self.context_builder.build_context(
            query=query_graphrag,
            conversation_history=None,
            **kwargs,
            **self.context_builder_params,
        )
        
        context_text = context_text.encode("raw_unicode_escape").decode('unicode-escape')
        log.info("GENERATE ANSWER: %s. QUERY: %s", start_time, query)
        try:
            search_prompt = self.system_prompt.format(
                context_data=context_text, response_type=self.response_type
            )
            print("SEARCH PRMOPT: %s", search_prompt)
            search_messages = [
                {"role": "system", "content": search_prompt},
                {"role": "user", "content": query},
            ]

            response = await self.llm.agenerate(
                messages=search_messages,
                streaming=True,
                callbacks=self.callbacks,
                **self.llm_params,
            )
            print("RESPONSE: %s", response)

            return SearchResult(
                response=response,
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
            )

        except Exception:
            log.exception("Exception in _asearch")
            return SearchResult(
                response="",
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
            )

    def get_context(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> tuple[str, dict[str, pd.DataFrame]]:
        """Only build local search context that fits a single context window."""

        if conversation_history:
            query_graphrag = self.refine_with_conversation_history(query, conversation_history)
        else:
            query_graphrag = query

        context_text, context_records = self.context_builder.build_context(
            query=query_graphrag,
            conversation_history=None,
            **kwargs,
            **self.context_builder_params,
        )

        try:
            context_text = context_text.encode("raw_unicode_escape").decode('unicode-escape')
        except:
            pass
        return context_text, context_records

    def search_with_context(
        self,
        query: str,
        context_text: str,
        context_records: dict[str, pd.DataFrame],
        callback: BaseLLMCallback,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> SearchResult:
        start_time = time.time()
        search_prompt = ""
        
        log.info("GENERATE ANSWER: %d. QUERY: %s", start_time, query)
        try:
            search_prompt = self.system_prompt.format(
                context_data=context_text, response_type=self.response_type
            )
            print("SEARCH PRMOPT: %s", search_prompt)
            search_messages = [
                {"role": "system", "content": search_prompt},
                {"role": "user", "content": query},
            ]
            # llm_execute()
            response = self.llm.generate(
                messages=search_messages,
                streaming=True,
                callbacks=[callback],
                **self.llm_params,
            )

            # 后处理去除重复内容
            # if response[-1] != "。":
            #     response = "。".join(response.split("。")[:-1])
            # response = remove_duplicate_sentences(response)
            print("RESPONSE: %s", response)

            return SearchResult(
                response=response,
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
            )

        except Exception:
            log.exception("Exception in _map_response_single_batch")
            return SearchResult(
                response="",
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
            )


    def refine_with_conversation_history(
        self,
        query: str,
        conversation_history: ConversationHistory,
    ) -> str:
        (
            conversation_history_context,
            _,
        ) = conversation_history.build_context(
            include_user_turns_only=False,
            recency_bias=True,
        )
        refine_prompt = REFINE_PROMPT.format(
        history_conversations=conversation_history_context
        )
        print("REFINE PRMOPT: %s", refine_prompt)
        search_messages = [
            {"role": "system", "content": refine_prompt},
            {"role": "user", "content": f"请基于对话历史，改写这个问题: {query}"},
        ]
        # llm_execute()
        response = self.llm.generate(
            messages=search_messages,
            streaming=True,
            callbacks=self.callbacks,
            **self.llm_params,
        )
        print(response)
        return response


    def search(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> SearchResult:
        """Build local search context that fits a single context window and generate answer for the user question."""

        """
        local_search:
            1. 对构建索引部分的实体构建向量，便于根据query去查询 
            2. 根据query检索出相关的topk的实体，每个实体与社区信息，协变量，实体关系等是存在对应关系的。
            3. 构建社区信息，实体信息，协变量信息，实体关系信息的相关数据
            4. 填充prompt模板
            5. 调用大模型
            6. 结果解析并返回
        """
        start_time = time.time()
        search_prompt = ""

        """
        if conversation_history:
            query_graphrag = self.refine_with_conversation_history(query, conversation_history)
        else:
            query_graphrag = query

        context_text, context_records = self.context_builder.build_context(
            query=query_graphrag,
            conversation_history=None,
            **kwargs,
            **self.context_builder_params,
        )
        """
        context_text, context_records = self.context_builder.build_context(
            query=query,
            conversation_history=conversation_history,
            **kwargs,
            **self.context_builder_params,
        )

        try:
            context_text = context_text.encode("raw_unicode_escape").decode('unicode-escape')
        except:
            pass
        print(context_text)
        log.info("GENERATE ANSWER: %d. QUERY: %s", start_time, query)
        try:
            search_prompt = self.system_prompt.format(
                context_data=context_text, response_type=self.response_type
            )
            print("SEARCH PRMOPT: %s", search_prompt)
            search_messages = [
                {"role": "system", "content": search_prompt},
                {"role": "user", "content": query},
            ]
            # llm_execute()
            response = self.llm.generate(
                messages=search_messages,
                streaming=True,
                callbacks=self.callbacks,
                **self.llm_params,
            )

            # 后处理去除重复内容
            # if response[-1] != "。":
            #     response = "。".join(response.split("。")[:-1])
            # response = remove_duplicate_sentences(response)
            print("RESPONSE: %s", response)

            return SearchResult(
                response=response,
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
            )

        except Exception:
            log.exception("Exception in _map_response_single_batch")
            return SearchResult(
                response="",
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
            )

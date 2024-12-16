# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 拆分text成text units

"""A module containing run and split_text_on_tokens methods definition."""

from collections.abc import Iterable
from typing import Any

import tiktoken
from datashaper import ProgressTicker

import graphrag.config.defaults as defs
from graphrag.index.text_splitting import Tokenizer
from graphrag.index.verbs.text.chunk.typing import TextChunk
from langchain_text_splitters import RecursiveCharacterTextSplitter


def run(
    input: list[str], args: dict[str, Any], tick: ProgressTicker
) -> Iterable[TextChunk]:
    """Chunks text into multiple parts. A pipeline verb."""
    tmp = [text.split("|title_split|") for text in input]
    input = [i[1] for i in tmp]
    title = [i[0] for i in tmp]
    tokens_per_chunk = args.get("chunk_size", defs.CHUNK_SIZE)
    chunk_overlap = args.get("chunk_overlap", defs.CHUNK_OVERLAP)
    encoding_name = args.get("encoding_name", defs.ENCODING_MODEL)
    enc = tiktoken.get_encoding(encoding_name)

    def encode(text: str) -> list[int]:
        if not isinstance(text, str):
            text = f"{text}"
        return enc.encode(text)

    def decode(tokens: list[int]) -> str:
        return enc.decode(tokens)

    return split_text_on_tokens(
        input,
        Tokenizer(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=tokens_per_chunk,
            encode=encode,
            decode=decode,
        ),
        tick,
        title,
    )


# Adapted from - https://github.com/langchain-ai/langchain/blob/77b359edf5df0d37ef0d539f678cf64f5557cb54/libs/langchain/langchain/text_splitter.py#L471
# So we could have better control over the chunking process
def split_text_on_tokens(
    texts: list[str], enc: Tokenizer, tick: ProgressTicker, title: list[str],
) -> list[TextChunk]:
    """Split incoming text and return chunks."""
    result = []
    mapped_ids = []

    for source_doc_idx, text in enumerate(texts):
        encoded = enc.encode(text)
        tick(1)
        mapped_ids.append((source_doc_idx, encoded))

    input_ids: list[tuple[int, int]] = [
        (source_doc_idx, id) for source_doc_idx, ids in mapped_ids for id in ids
    ]

    start_idx = 0
    cur_idx = min(start_idx + enc.tokens_per_chunk, len(input_ids))
    if cur_idx < len(input_ids):
        while input_ids[cur_idx-1][1] != enc.encode("。")[0]:
            cur_idx += 1
    chunk_ids = input_ids[start_idx:cur_idx]
    while start_idx < len(input_ids):
        chunk_text = title[source_doc_idx] + "\n" + enc.decode([id for _, id in chunk_ids])
        doc_indices = list({doc_idx for doc_idx, _ in chunk_ids})
        result.append(
            TextChunk(
                text_chunk=chunk_text,
                source_doc_indices=doc_indices,
                n_tokens=len(chunk_ids),
            )
        )
        if cur_idx == len(input_ids):
            break
        start_idx = cur_idx - enc.chunk_overlap
        cur_idx = min(start_idx + enc.tokens_per_chunk, len(input_ids))
        if cur_idx < len(input_ids):
            while input_ids[cur_idx+1][1] != enc.encode("。")[0]:
                cur_idx += 1
                if cur_idx == len(input_ids)-1:
                    cur_idx = len(input_ids)
                    break
        while input_ids[start_idx][1] != enc.encode("。")[0]:
            start_idx -= 1
        start_idx += 1
        chunk_ids = input_ids[start_idx:cur_idx]
    return result

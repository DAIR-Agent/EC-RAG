from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
import argparse
# from langchain_ollama import OllamaEmbeddings
from pathlib import Path
import os
import re
from graphrag.config import (
    GraphRagConfig,
    create_graphrag_config,
)
from graphrag.query.factories import get_llm

import re

def remove_after_last_punctuation(text):
    # 使用正则表达式寻找最后一个标点符号
    match = re.search(r'[。！？.!?]', text[::-1])
    
    if match:
        # 根据匹配到的位置切割字符串，去掉最后一个标点及其后面的部分
        cut_position = len(text) - match.start()
        return text[:cut_position+1]
    
    # 如果没有标点符号，则返回原字符串
    return text

def has_duplicate_sentence(text):
    # 使用正则表达式将文本分割为句子
    sentences = re.split(r'[。！？.!?]', text)
    
    # 去掉空的句子
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    # 使用集合来判断是否有重复的句子
    seen = set()
    for sentence in sentences:
        if sentence in seen:
            return True  # 找到重复的句子
        seen.add(sentence)
    
    return False  # 没有找到重复的句子

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='root path', required=True)
    parser.add_argument('--output', help='root path', required=True)

    args = parser.parse_args()

    root = args.root
    outdir = args.output

    DEFAULT_LLM_PARAMS = {
    "max_tokens": 1500,
    "temperature": 0.0,
    }
    llm_params_RAG = DEFAULT_LLM_PARAMS

    # LLM
    _root = Path(root)
    settings_yaml = _root / "settings.yaml"
    if not settings_yaml.exists():
        settings_yaml = _root / "settings.yml"

    if settings_yaml.exists():
        print(f"Reading settings from {settings_yaml}")
        with settings_yaml.open("r") as file:
            import yaml
            data = yaml.safe_load(file)
            config = create_graphrag_config(data, root)
    else:
        print("Reading settings from environment variables")
        config = create_graphrag_config(root_dir=root)

    llm_RAG_summary = get_llm(config)

    # Load Documents
    path = f"./{root}/input"

    file_list = os.listdir(path)

    PROMPT = "---角色---\
    \
    你是一名智能助手，根据这段文字，提供一个简短的总结，不超过200字。请注意这不是一个补齐任务。\n\
    {context} \n\
    ---目标---\n\
    生成一个符合目标长度和格式的回应，回答用户的问题。"

    PROMPT2 = "---角色---\
    \
    你是一名智能助手，如果这段文字中存在重复内容，请去除重复重新整理后输出，否则就直接输出这段文字。\n\
    {context} \n\
    ---目标---\n\
    生成一个符合目标长度和格式的回应，回答用户的问题。"

    docs = []
    for filename in file_list:
        file_path = f"{path}/{filename}"
        
        with open(file_path, "r") as f:
            content = f.read()
            
        chapter = content.split(" ")[0]

        print(f"start processing {chapter}")

        book_name = re.findall(r'\D+', filename)[0].split(".")[0].replace(chapter, "")

        prompt = PROMPT.format(context=content)
        
        search_messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "请生成一段简洁的文字来总结一下文本中的内容。"},
        ]
        # llm_execute()
        result = llm_RAG_summary.generate(
                        messages=search_messages,
                        streaming=False,
                        **llm_params_RAG,
                    )

        # print(result)
        
        cnt = 0
        while True:
            cnt += 1
            print(cnt)

            if has_duplicate_sentence(result):
                print("There is duplication")
                result = remove_after_last_punctuation(result)
                print(result)
                prompt2 = PROMPT2.format(context=result)
                
                search_messages2 = [
                    {"role": "system", "content": prompt2},
                    {"role": "user", "content": "如果这段文字中存在重复内容，请去除重复重新整理后输出，否则就直接输出这段文字，输出中不要有重复的句子。"},
                ]
                # llm_execute()
                result = llm_RAG_summary.generate(
                                messages=search_messages2,
                                streaming=False,
                                **llm_params_RAG,
                            )
                if len(result) == 0:
                    continue
                result = remove_after_last_punctuation(result)
                break
            else:
                if len(result) > 500:
                    if cnt <= 5:
                        result = llm_RAG_summary.generate(
                                    messages=search_messages,
                                    streaming=False,
                                    **llm_params_RAG,
                                )
                    else:
                        prompt = PROMPT.format(context=result)
        
                        search_messages = [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": "请生成一段简洁的文字来总结一下文本中的内容。"},
                        ]
                        # llm_execute()
                        result = llm_RAG_summary.generate(
                                        messages=search_messages,
                                        streaming=False,
                                        **llm_params_RAG,
                                    )
                        if len(result) == 0:
                            break
                        result = remove_after_last_punctuation(result)
                        break
                else:
                    break
                    

        # print(result)

        print(book_name+chapter+"\n"+result+"\n"+book_name+chapter)

        document = Document(
            page_content=book_name+chapter+"\n"+result+"\n"+book_name+chapter,
            metadata={"来源": book_name+chapter}
        )
        docs.append(document)
        print(f"finished processing {chapter}")
    
    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # Embed
    
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    hf = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3")

    model_name = "BAAI/bge-m3"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity

    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction="为这个段落生成表示以用于检索相关文章："
    )

    """
    hf = OllamaEmbeddings(
        model="bge-m3"
    )
    """

    vectorstore = Chroma.from_documents(documents=splits, embedding=hf, persist_directory=f"{root}/chroma_langchain_db/{outdir}_summary")
    print("Successfully finished index")


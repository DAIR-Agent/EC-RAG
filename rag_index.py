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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='root path', required=True)
    parser.add_argument('--output', help='root path', required=True)

    args = parser.parse_args()

    root = args.root
    outdir = args.output

    # Load Documents
    path = f"./{root}/input"

    file_list = os.listdir(path)

    docs = []
    for filename in file_list:
        file_path = f"{path}/{filename}"
        
        with open(file_path, "r") as f:
            content = f.read()
            
        chapter = content.split(" ")[0]

        book_name = re.findall(r'\D+', filename)[0].split(".")[0].replace(chapter, "")

        print(book_name+chapter)

        document = Document(
            page_content=content,
            metadata={"来源": book_name+chapter}
        )
        docs.append(document)

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

    vectorstore = Chroma.from_documents(documents=splits, embedding=hf, persist_directory=f"{root}/chroma_langchain_db/{outdir}")
    print("Successfully finished index")


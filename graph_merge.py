import os
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
import chromadb

from chromadb import Documents, EmbeddingFunction, Embeddings
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from graphrag.query.factories import get_llm
from graphrag.config import (
    create_graphrag_config,
)
from pathlib import Path


class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, hf):
        self.hf = hf
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        return self.hf.embed_documents(input)

def map_type(map):
    map_dict = {"人物": "Person", "地点": "Location", "事件": "Event"}
    return map_dict.get(map, map) 

def map_type_reverse(map):
    map_dict = {"Person": "人物", "Location": "地点", "Event": "事件"}
    return map_dict.get(map, map) 

def query_neo4j(driver, query):
    """Executes a query in Neo4j and returns the results."""
    with driver.session() as session:
        result = session.run(query)
        return [record for record in result]

def llm_verify(llm, llm_params, prompt):
    search_messages = [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": "请根据数据表判断实体A和实体B是否为同一实体"},
                        ]
    # print(search_messages)
    response = result = llm.generate(
            messages=search_messages,
            streaming=False,
            **llm_params,
        )
    return response

def llm_merge(llm, llm_params, prompt):
    search_messages = [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": "请请文本中所有这些描述整合为一个单一且全面的描述"},
                        ]
    # print(search_messages)
    response = llm.generate(
            messages=search_messages,
            streaming=False,
            **llm_params,
        )
    return response

GRAPHRAG_FOLDER = "./ragtest_test2/output/hongloumeng2/artifacts"
NEO4J_URI = "neo4j://localhost"  # or neo4j+s://xxxx.databases.neo4j.io
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "12345678" #你自己的密码
NEO4J_DATABASE = "neo4j"

root = "./ragtest_test2"
_root = Path(root)
settings_yaml = _root / "settings.yaml"

if settings_yaml.exists():
    print(f"Reading settings from {settings_yaml}")
    with settings_yaml.open("r") as file:
        import yaml
        data = yaml.safe_load(file)
        config = create_graphrag_config(data, root)

llm = get_llm(config)
DEFAULT_LLM_PARAMS = {
"max_tokens": 1500,
"temperature": 0.0,
}
llm_params = DEFAULT_LLM_PARAMS

PROMPT =  """
---角色---

你是一名智能助手，使用数据表中的数据回答用户的问题。你除了数据表中的数据没有任何额外信息来源。

---规则---

只回答是或者否。

----数据表---

{0}

"""

PROMPT2=  """
---角色---

你是一名智能助手，使用文本回答用户的问题。

---规则---

给定文本，请将所有这些描述去除重复内容后整合为一个单一且全面的描述，确保包含所有描述中的信息，不添加额外信息。
不要简单地将描述合并在一起，而是整理一下再合并避免出现重复的内容。
如果提供的描述有矛盾，请解决矛盾，并提供一个连贯的总结。
结果中如果有重复的语句，请重新组织合并后再输出。
请确保使用第三人称撰写，并包含实体名称，以便我们有完整的上下文。
请不要使用你的内置知识来回答这个问题。
请只从我给定的数据中获取信息，请不要加入提供的数据以外的内容。
请不要编造内容。
请不要加入额外的内容。

---文本---

{0}

"""

# Create a Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

entity_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/create_final_entities_ori.parquet',
                            columns=["name", "type", "description", "human_readable_id", "id", "description_embedding",
                                     "text_unit_ids"])
rel_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/create_final_relationships_ori.parquet',
                         columns=["source", "target", "id", "rank", "weight", "human_readable_id", "description",
                                  "text_unit_ids", "source_degree", "target_degree"])
# text_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/create_final_text_units.parquet',
#                          columns=["id","text","n_tokens","document_ids"])

# community_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/create_final_communities.parquet')
# community_reports_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/create_final_community_reports.parquet')

# text_unit_dict = {}
# for i in range(len(text_df)):
#    text_unit_dict[text_df["id"].iloc[i]] = text_df["text"].iloc[i]

entity_df["name"] = entity_df["name"].apply(lambda x: x[1:-1])
entity_df["type"] = entity_df["type"].apply(lambda x: x[1:-1] if x is not None else x)
entity_df["description"] = entity_df["description"].apply(lambda x: x[1:-1] if (x is not None) and (x[0] == '"') and (x[-1] == '"') else x)
rel_df["source"] = rel_df["source"].apply(lambda x: x[1:-1])
rel_df["target"] = rel_df["target"].apply(lambda x: x[1:-1])
rel_df["description"] = rel_df["description"].apply(lambda x: x[1:-1] if (x[0] == '"') and (x[-1] == '"') else x)

neo4j_query = f"""
MATCH (n:__Entity__)
RETURN n.name,n.info,labels(n)
"""

results = query_neo4j(driver, neo4j_query)


name_list = []
info_list = []
entity_df_ori = pd.DataFrame(columns=entity_df.columns)
for i in range(len(results)):
    record = results[i]
    name = record['n.name']
    info = record['n.info']
    type = [i for i in record['labels(n)'] if i != "__Entity__"][0]
    name_list.append(name)
    info_list.append(info)
    entity_df_ori.loc[i] = [name, type, info, None, None, None, []]


neo4j_query = f"""
MATCH (n:__Entity__)-[r:RELATED]->(m:__Entity__)
RETURN n.name, labels(n), r.label, m.name, labels(m)
"""
results_rel = query_neo4j(driver, neo4j_query)

rel_df_ori = pd.DataFrame(columns=rel_df.columns)
for i in range(len(results_rel)):
    record = results_rel[i]
    name1 = record['n.name']
    label = record['r.label']
    name2 = record['m.name']
    type_n = [i for i in record['labels(n)'] if i != "__Entity__"][0]
    type_m = [i for i in record['labels(m)'] if i != "__Entity__"][0]
    if type_n == "Person" and type_m == "Person":
        label = f"{name2}是{name1}的{record['r.label']}"
    elif type_n == "Person" and type_m == "Event":
        label = f"{name1}{record['r.label']}{name2}"
    elif type_n == "Person" and type_m == "Location":
        label = f"{name2}是{name1}的{record['r.label']}"
    elif type_n == "Location" and type_m == "Location":
        label = f"{name1}{record['r.label']}{name2}"
    rel_df_ori.loc[i] = [name1, name2, None, None, None, None, label, [], None, None]

root = "./ragtest_test2"
outdir = "hongloumeng"

# model = SentenceTransformer('./all-MiniLM-L6-v2')

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# hf = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3")

model_name = "BAAI/bge-m3"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity

hf = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个段落生成表示以用于检索"
)

chroma_client = chromadb.PersistentClient(path=f"{root}/chroma_langchain_db/{outdir}_merge")

try:
    chroma_client.delete_collection(name="name")
except:
    pass

collection_name = chroma_client.create_collection(name="name", embedding_function=MyEmbeddingFunction(hf), metadata={"hnsw:space": "ip"})
collection_name.add(
    documents=name_list,
    ids=name_list
)

try:
    chroma_client.delete_collection(name="description")
except:
    pass

collection_info = chroma_client.create_collection(name="description", embedding_function=MyEmbeddingFunction(hf), metadata={"hnsw:space": "ip"})
info_new_list = []
name_new_list = []
for i in range(len(info_list)):
    if info_list[i] is not None:
        info_new_list.append(info_list[i])
        name_new_list.append(name_list[i])

collection_info.add(
    documents=info_new_list,
    ids=name_new_list
)


embedding_dict_ori = {}
for i in range(len(entity_df_ori)):
    entity = entity_df_ori["name"].iloc[i]
    df_rel_tmp1 = rel_df_ori[rel_df_ori["source"]==entity]
    df_rel_tmp2 = rel_df_ori[rel_df_ori["target"]==entity]
    embedding1 = [hf.embed_query(i) for i in df_rel_tmp1["target"].values]
    embedding2 = [hf.embed_query(i) for i in df_rel_tmp2["source"].values]
    embedding = np.sum(np.array(embedding1), axis=0) + np.sum(np.array(embedding2), axis=0) + hf.embed_query(entity)
    embedding_dict_ori[entity] = embedding


embedding_dict = {}
for i in range(len(entity_df)):
    entity = entity_df["name"].iloc[i]
    df_rel_tmp1 = rel_df[rel_df["source"]==entity]
    df_rel_tmp2 = rel_df[rel_df["target"]==entity]
    embedding1 = [hf.embed_query(i) for i in df_rel_tmp1["target"].values]
    embedding2 = [hf.embed_query(i) for i in df_rel_tmp2["source"].values]
    embedding = np.sum(np.array(embedding1), axis=0) + np.sum(np.array(embedding2), axis=0) + hf.embed_query(entity)
    embedding_dict[entity] = embedding




merge_dict = {}

for i in range(len(entity_df)):
    name = entity_df["name"].iloc[i]
    description = entity_df["description"].iloc[i]
    type = entity_df["type"].iloc[i]
    if description is None and type is None:
        continue
    type = map_type(type)
    result = collection_name.query(
            query_texts=name,
            n_results=collection_name.count()
            )
    if name in entity_df_ori["name"].values:
        print(f"Merge {name} with {name}")
        continue
    score1_dict = {}
    for j in range(len(result["ids"][0])):
        score = 1-result["distances"][0][j]
        score1_dict[result["ids"][0][j]] = score
    result = collection_info.query(
            query_texts=description,
            n_results=collection_info.count()
            )
    score2_dict = {}
    description_dict = {}
    for j in range(len(result["ids"][0])):
        score = 1-result["distances"][0][j]
        score2_dict[result["ids"][0][j]] = score
        description_dict[result["ids"][0][j]] = result["documents"][0][j]
    merge_target_list = []
    corr_list = []
    overlap_list = []
    description_list = []
    for merge_target in score1_dict.keys():
        if isinstance(embedding_dict_ori[merge_target], float):
            continue
        if isinstance(embedding_dict[name], float):
            continue
        embedding = embedding_dict[name]
        embedding_ori = embedding_dict_ori[merge_target]
        corr = embedding @ embedding_ori.T / np.sqrt(embedding @ embedding.T) / np.sqrt(embedding_ori @ embedding_ori.T)
        # df_rel_tmp1 = rel_df[(rel_df["source"]==name)&(rel_df["target"]==merge_target)&(~rel_df['description'].str.contains("相同", na=False))]
        # df_rel_tmp2 = rel_df[(rel_df["source"]==merge_target)&(rel_df["target"]==name)&(~rel_df['description'].str.contains("相同", na=False))]
        # df_rel_tmp3 = rel_df_ori[(rel_df_ori["source"]==name)&(rel_df_ori["target"]==merge_target)]
        # df_rel_tmp4 = rel_df_ori[(rel_df_ori["source"]==merge_target)&(rel_df_ori["target"]==name)]
        type_merge_target = entity_df_ori[entity_df_ori["name"] == merge_target]["type"].iloc[0]
        # if len(df_rel_tmp1) + len(df_rel_tmp2) + len(df_rel_tmp3) + len(df_rel_tmp4) == 0 and type == type_merge_target:
        if type == type_merge_target:
            # overlap = len(set(name).intersection(set(merge_target))) > 0
            overlap = len(set(name).intersection(set(merge_target)))
            value = corr+score2_dict.get(merge_target, 0)
            if value > 1.58:
                merge_target_list.append(merge_target)
                overlap_list.append(overlap)
                corr_list.append(value)
                description_list.append(description_dict.get(merge_target, ""))
    if len(corr_list) == 0:
        print(f"Didn't merge {name}")
        continue
    max_corr = corr_list[0]
    overlap_max = max(overlap_list)
    max_index = 0
    # print(corr_list)
    # print(merge_target_list)
    for i in range(len(corr_list)):
        if corr_list[i] > max_corr and overlap_list[i]>=overlap_max:
            max_corr = corr_list[i]
            max_index = i
    merge_target = merge_target_list[max_index]
    neo4j_query = f"""
    MATCH (n:__Entity__)-[r:RELATED]->(m:__Entity__)
    WHERE '{merge_target}' = n.name or '{merge_target}' = m.name
    RETURN n.name, labels(n), r.label, r.id, m.name, labels(m)
    """
    results = query_neo4j(driver, neo4j_query)
    context = ""
    for record in results:
        name_n = record['n.name']
        type_n = [i for i in record['labels(n)'] if i != "__Entity__"][0]
        name_m = record['m.name']
        type_m = [i for i in record['labels(m)'] if i != "__Entity__"][0]
        if type_n == "Person" and type_m == "Person":
            context += f"{name_m}是{name_n}的{record['r.label']} \n"
        elif type_n == "Person" and type_m == "Event":
            context += f"{name_n}{record['r.label']}{name_m} \n"
        elif type_n == "Person" and type_m == "Location":
            context += f"{name_m}是{name_n}的{record['r.label']} \n"
        elif type_n == "Location" and type_m == "Location":
            context += f"{name_n}{record['r.label']}{name_m} \n"
    description = description_dict.get(merge_target, "")
    information1 = f"实体A\n 实体A名称：{merge_target}\n 实体A描述：{description} \n 实体A关系：{context}" 
    df_rel_tmp1 = rel_df[rel_df["source"]==name]
    df_rel_tmp2 = rel_df[rel_df["target"]==name]
    context = ""
    for j in range(len(df_rel_tmp1)):
        description = df_rel_tmp1["description"].iloc[j]
        context += f"{description} \n"
    for j in range(len(df_rel_tmp2)):
        description = df_rel_tmp2["description"].iloc[j]
        context += f"{description} \n"
    information2 = f"实体B\n 实体B名称：{name}\n 实体B描述：{description} \n \
                    实体B关系：{context}"
    # text_unit_ids = entity_df["text_unit_ids"].iloc[i]
    # text_unit_text = ""
    # for j in range(len(text_unit_ids)):
    #    text_unit_text += f"{text_unit_dict[text_unit_ids[j]]}\n"
    # result = llm_verify(llm, llm_params, PROMPT.format(information1 + "\n\n\n" + information2 + "原文内容： \n" + text_unit_text[:100000]))
    result = llm_verify(llm, llm_params, PROMPT.format(information1 + "\n\n\n" + information2))
    if "否" in result:
        print(f"Didn't merge {name}")
        pass
    elif "是" in result:
        print(f"Merge {name} with {[merge_target_list[max_index]]}, {corr_list[max_index]}")
        merge_dict[name] = merge_target_list[max_index]
assert(False)
for name in merge_dict.keys():
    entity_df.loc[entity_df["name"]==name, "name"] = merge_dict[name]
    rel_df.loc[rel_df["source"]==name, "source"] = merge_dict[name]
    rel_df.loc[rel_df["target"]==name, "target"] = merge_dict[name]

entity_df_ori["description"] = entity_df_ori["description"].apply(lambda x: [x] if x is not None else [])
rel_df_ori["description"] = rel_df_ori["description"].apply(lambda x: [x] if x is not None else [])

entity_check_dict = {}
for i in range(len(entity_df)):
    name = entity_df["name"].iloc[i]
    description = entity_df["description"].iloc[i]
    type = entity_df["type"].iloc[i]
    text_unit_ids = entity_df["text_unit_ids"].iloc[i]
    id = entity_df["id"].iloc[i]
    human_readable_id = entity_df["human_readable_id"].iloc[i]
    description_embedding = entity_df["description_embedding"].iloc[i]
    if name in entity_df_ori["name"].values:
        # 把entity描述加入原知识图谱的实体中
        entity_df_ori.loc[entity_df_ori["name"]==name, "description"] = entity_df_ori.loc[entity_df_ori["name"]==name, "description"].apply(lambda x: x+[description])
        entity_df_ori.loc[entity_df_ori["name"]==name, "description_embedding"] = entity_df_ori.loc[entity_df_ori["name"]==name, "description_embedding"].apply(lambda x: description_embedding)
        entity_df_ori.loc[entity_df_ori["name"]==name, "text_unit_ids"] = entity_df_ori.loc[entity_df_ori["name"]==name, "text_unit_ids"].apply(lambda x: list(x)+list(text_unit_ids))
        if entity_check_dict.get(name, 0):
            continue
        # 把改entity所有自己from的关系加入知识图谱中
        entity_df_ori.loc[entity_df_ori["name"]==name, "id"] = id
        entity_df_ori.loc[entity_df_ori["name"]==name, "human_readable_id"] = human_readable_id
        rel_df_tmp = rel_df[rel_df["source"]==name]
        for j in range(len(rel_df_tmp)):
            target = rel_df_tmp["target"].iloc[j]
            description = rel_df_tmp["description"].iloc[j]
            id = rel_df_tmp["id"].iloc[j]
            rank = rel_df_tmp["rank"].iloc[j]
            weight = rel_df_tmp["weight"].iloc[j]
            human_readable_id = rel_df_tmp["human_readable_id"].iloc[j]
            text_unit_ids = rel_df_tmp["text_unit_ids"].iloc[j]
            source_degree = rel_df_tmp["source_degree"].iloc[j]
            target_degree = rel_df_tmp["target_degree"].iloc[j]
            rel_df_ori_tmp = rel_df_ori[(rel_df_ori["source"]==name)&(rel_df_ori["target"]==target)]
            if description is not None:
                if len(rel_df_ori_tmp) > 0:
                    rel_df_ori.loc[(rel_df_ori["source"]==name)&(rel_df_ori["target"]==target), "description"] = rel_df_ori.loc[(rel_df_ori["source"]==name)&(rel_df_ori["target"]==target), "description"].apply(lambda x: x+[description])
                    rel_df_ori.loc[(rel_df_ori["source"]==name)&(rel_df_ori["target"]==target), "rank"] = rank
                    rel_df_ori.loc[(rel_df_ori["source"]==name)&(rel_df_ori["target"]==target), "weight"] = weight
                    rel_df_ori.loc[(rel_df_ori["source"]==name)&(rel_df_ori["target"]==target), "id"] = id
                    rel_df_ori.loc[(rel_df_ori["source"]==name)&(rel_df_ori["target"]==target), "source_degree"] = source_degree
                    rel_df_ori.loc[(rel_df_ori["source"]==name)&(rel_df_ori["target"]==target), "target_degree"] = target_degree
                    rel_df_ori.loc[(rel_df_ori["source"]==name)&(rel_df_ori["target"]==target), "human_readable_id"] = human_readable_id
                    rel_df_ori.loc[(rel_df_ori["source"]==name)&(rel_df_ori["target"]==target), "text_unit_ids"] = rel_df_ori.loc[(rel_df_ori["source"]==name)&(rel_df_ori["target"]==target), "text_unit_ids"].apply(lambda x: list(x)+list(text_unit_ids))
                else:
                    rel_df_ori.loc[len(rel_df_ori)] = [name, target, id, rank, weight, human_readable_id, [description], text_unit_ids, source_degree, target_degree]
        entity_check_dict[name] = 1
    else:
        # 把entity描述加入原知识图谱的实体中
        entity_df_ori.loc[len(entity_df_ori)] = [name, map_type(type), [description], human_readable_id, id, description_embedding, text_unit_ids]
        rel_df_tmp = rel_df[rel_df["source"]==name]
        if entity_check_dict.get(name, 0):
            continue
        # 把改entity所有自己from的关系加入知识图谱中
        for j in range(len(rel_df_tmp)):
            target = rel_df_tmp["target"].iloc[j]
            description = rel_df_tmp["description"].iloc[j]
            id = rel_df_tmp["id"].iloc[j]
            rank = rel_df_tmp["rank"].iloc[j]
            weight = rel_df_tmp["weight"].iloc[j]
            human_readable_id = rel_df_tmp["human_readable_id"].iloc[j]
            text_unit_ids = rel_df_tmp["text_unit_ids"].iloc[j]
            source_degree = rel_df_tmp["source_degree"].iloc[j]
            target_degree = rel_df_tmp["target_degree"].iloc[j]
            if description is not None:
                rel_df_ori.loc[len(rel_df_ori)] = [name, target, id, rank, weight, human_readable_id, [description], text_unit_ids, source_degree, target_degree]
        entity_check_dict[name] = 1

for i in range(len(entity_df_ori)):
    index = entity_df_ori.index[i]
    description_list = [j for j in entity_df_ori.loc[index, "description"] if j is not None][::-1]
    description = " ".join(description_list)
    if len(description_list)<=1:
       entity_df_ori.loc[index, "description"] = description
    else:
        # print(description_list)
        entity_df_ori.loc[index, "description"] = llm_merge(llm, llm_params, PROMPT2.format(description))
        print(entity_df_ori.loc[index, "description"])

for i in range(len(rel_df_ori)):
    index = rel_df_ori.index[i]
    description_list = [j for j in rel_df_ori.loc[index, "description"] if j is not None][::-1]
    description = " ".join(description_list)
    if len(description_list)<=1:
       rel_df_ori.loc[index, "description"] = description
    else:
        # print(description_list)
        rel_df_ori.loc[index, "description"] = llm_merge(llm, llm_params, PROMPT2.format(description))
        print(rel_df_ori.loc[index, "description"])

### 重新进行embedding
entity_df_ori["description"] =  entity_df_ori["name"] + ":"+ entity_df_ori["description"]
entity_df_ori["description_embedding"] = [hf.embed_query(entity_df_ori["description"].iloc[i]) for i in range(len(entity_df_ori))]

### 重新存储到neo4j数据库


### 存储到create_final_entities.parquet和create_final_relationships.parquet
entity_df_ori["type"] = entity_df_ori["type"].apply(map_type_reverse)
entity_df_ori["graph_embedding"] = None
entity_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/create_final_entities_ori.parquet')
entity_df_ori = entity_df_ori.reindex(columns=entity_df.columns)
ret_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/create_final_relationships_ori.parquet')
rel_df_ori = rel_df_ori.reindex(columns=ret_df.columns)
entity_df_ori["name"] = entity_df_ori["name"].apply(lambda x: f'"{x}"')
entity_df_ori["name_embedding"] = [hf.embed_query(entity_df_ori["name"].iloc[i]) for i in range(len(entity_df_ori))]
entity_df_ori["type"] = entity_df_ori["type"].apply(lambda x: f'"{x}"')
entity_df_ori["description"] = entity_df_ori["description"].apply(lambda x: f'"{x}"')
rel_df_ori["source"] = rel_df_ori["source"].apply(lambda x: f'"{x}"')
rel_df_ori["target"] = rel_df_ori["target"].apply(lambda x: f'"{x}"')
rel_df_ori["description"] = rel_df_ori["description"].apply(lambda x: f'"{x}"')
entity_df_ori.to_parquet(f'{GRAPHRAG_FOLDER}/create_final_entities.parquet')
rel_df_ori.to_parquet(f'{GRAPHRAG_FOLDER}/create_final_relationships.parquet')



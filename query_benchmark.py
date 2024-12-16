from neo4j import GraphDatabase
from graphrag.query.factories import get_llm
from graphrag.config import (
    create_graphrag_config,
)
from pathlib import Path

root = "./ragtest_test2"
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

# Connect to Neo4j database
uri = "bolt://localhost:7687"
username = "neo4j"
password = "12345678"
driver = GraphDatabase.driver(uri, auth=(username, password))

query = "贾宝玉和贾敬的关系"

SEARCH_SYSTEM_PROMPT = """
---角色---

你是一名智能助手，使用数据表中的数据回答用户的问题。你除了数据表中的数据没有任何额外信息来源。


----目标---

生成一个符合目标长度和格式的回应，回答用户的问题，摘要输入数据表中的所有相关信息，并整合任何相关的通用知识。

如果你不知道答案，只需直说。不要编造任何内容。

如果涉及到计次数的问题，请基于被数据支持的内容统计次数。

数据表是你唯一的信息来源，不要编造和猜测。

涉及到书或者文章的名字请根据数据表确定正确的书或者文章名，不要编造，如果名字不完整只是部分则请进行补全。

不要在回答中提到与社区之间的联系。

这不是补齐任务。

请确保回答问题的内容是与问题有关的，例如问题是为什么，答案不要讲原因之外的内容。

被数据支持的要点应按以下格式列出其数据参考：

“这是一个由多个数据参考支持的示例句子 [数据: <数据集名称> (记录ID); <数据集名称> (记录ID)]。”

在单个引用中不要列出超过5个记录ID。请列出最相关的前5个记录ID，并加上“+更多”以表示还有更多。

请确定引用与用户的问题有关，特别是实体名称。

例如：

“甲先生是乙公司的所有者，并受到多项不当行为指控 [数据: 来源 (15, 16), 报告 (1), 实体 (5, 7); 关系 (23); 声明 (2, 7, 34, 46, 64, +更多)]。”

其中15, 16, 1, 5, 7, 23, 2, 7, 34, 46和64代表相关数据记录的ID（而非索引）。

不包含没有提供支持证据的信息。

如果在数据表中没有提及请回答你不知道答案并且除此之外不要提供任何回答。

不要使用数据表以外的知识进行回答。

如果问题中的实体在数据表中没有出现，请回答你不知道。


---目标回答长度和格式---

多段落


---数据表---

{context_data}


---目标---

生成一个符合目标长度和格式的回应，回答用户的问题，摘要输入数据表中的所有相关信息，并整合任何相关的通用知识。

如果你不知道答案，只需直说。不要编造任何内容。

如果涉及到计次数的问题，请基于被数据支持的内容统计次数。

涉及到书或者文章的名字请根据数据表确定正确的书或者文章名，不要编造，如果名字不完整只是部分则请进行补全。

不要在回答中提到与社区之间的联系。

这不是补齐任务。

请确保回答问题的内容是与问题有关的，例如问题是为什么，答案不要讲原因之外的内容。

被数据支持的要点应按以下格式列出其数据参考：

“这是一个由多个数据参考支持的示例句子 [数据: <数据集名称> (记录ID); <数据集名称> (记录ID)]。”

在单个引用中不要列出超过5个记录ID。请列出最相关的前5个记录ID，并加上“+更多”以表示还有更多。

请确定引用与用户的问题有关，特别是实体名称。

例如：

“甲先生是乙公司的所有者，并受到多项不当行为指控 [数据: 来源 (15, 16), 报告 (1), 实体 (5, 7); 关系 (23); 声明 (2, 7, 34, 46, 64, +更多)]。”

其中15, 16, 1, 5, 7, 23, 2, 7, 34, 46和64代表相关数据记录的ID（而非索引）。

不包含没有提供支持证据的信息。

如果在数据表中没有明确提及请回答你不知道答案并且除此之外不要提供任何回答。

不要使用数据表以外的知识进行回答。

如果问题中的实体在数据表中没有出现，请回答你不知道。


---目标回答长度和格式---

多段落


根据回应的长度和格式，适当添加章节和评论。将回应样式设为Markdown格式。
"""

def query_neo4j(driver, query):
    """Executes a query in Neo4j and returns the results."""
    with driver.session() as session:
        result = session.run(query)
        return [record for record in result]

def generate_response_with_llm(query, context):
    """Uses OpenAI's LLM to generate a response based on the context and prompt."""
    llm_RAG = get_llm(config)
    DEFAULT_LLM_PARAMS = {
    "max_tokens": 1500,
    "temperature": 0.0,
    }
    llm_params_RAG = DEFAULT_LLM_PARAMS

    search_messages = [
                            {"role": "system", "content": context},
                            {"role": "user", "content": query},
                        ]

    print(search_messages)

    response = result = llm_RAG.generate(
            messages=search_messages,
            streaming=False,
            **llm_params_RAG,
        )
    return response


neo4j_query = f"""
MATCH (n:__Entity__)-[r:RELATED]->(m:__Entity__)
WHERE '{query}' CONTAINS n.name or '{query}' CONTAINS m.name
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
        context += f"{name_n}是{name_m}的{record['r.label']} 关系（{record['r.id']}）\n"
    elif type_n == "Person" and type_m == "Event":
        context += f"{name_n}{record['r.label']}{name_m} 关系（{record['r.id']}）\n"
    elif type_n == "Person" and type_m == "Location":
        context += f"{name_n}是{name_m}的{record['r.label']} 关系（{record['r.id']}）\n"
    elif type_n == "Location" and type_m == "Location":
        context += f"{name_n}{record['r.label']}{name_m} 关系（{record['r.id']}）\n"
    
neo4j_query = f"""
MATCH (n:__Entity__)
WHERE '{query}' CONTAINS n.name
RETURN n.name, n.info, n.id
"""

results = query_neo4j(driver, neo4j_query)

for record in results:
    context += f"{record['n.name']}: {record['n.info']} 实体（{record['n.id']}）\n"
     
context = SEARCH_SYSTEM_PROMPT.replace("{context_data}", context)

response = generate_response_with_llm(query, context)

# Print the response from the LLM
print("LLM Response:", response)

# Close the Neo4j driver
driver.close()
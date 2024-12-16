import pandas as pd
from graphrag.query.factories import get_llm
from graphrag.config import (
    create_graphrag_config,
)
from pathlib import Path

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

GRAPHRAG_FOLDER = "/data/train/users/wenluo/projects/GraphRAG-Ollama-UI/ragtest_test2/output/红楼梦/artifacts"

entity_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/create_final_entities.parquet',
                            columns=["name", "type", "description", "human_readable_id", "id", "description_embedding",
                                     "text_unit_ids"])

text_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/create_final_text_units.parquet',
                          columns=["id","text","n_tokens","document_ids"])

text_unit_dict = {}
for i in range(len(text_df)):
    text_unit_dict[text_df["id"].iloc[i]] = text_df["text"].iloc[i]

PROMPT =  """
---角色---

你是一名智能助手，使用数据表中的数据回答用户的问题。你除了数据表中的数据没有任何额外信息来源。

---规则---

不要提到数据表

----数据表---

{0}

"""

for i in range(len(entity_df)):
    if entity_df["name"].iloc[i] != f'"邢夫人"':
        continue
    description = entity_df["description"].iloc[i]
    text_unit_ids = entity_df["text_unit_ids"].iloc[i]
    text_unit_text = ""
    for j in range(len(text_unit_ids)):
        text_unit_text += f"{text_unit_dict[text_unit_ids[j]]}\n"
    prompt = PROMPT.format(text_unit_text)

    # response = generate_response_with_llm("请根据数据表中的内容对上面的描述解决其中的矛盾后输出", prompt)
    response = generate_response_with_llm("请根据数据表中的内容判断邢夫人和邢岫烟的关系", prompt)

    # Print the response from the LLM
    print("Description before:", description)
    print("LLM Response:", response)

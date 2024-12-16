# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A file containing prompts definition."""

SUMMARIZE_PROMPT = """
你是一名负责生成全面摘要的助手，根据以下提供的数据生成总结。
给定一个或两个实体，以及与相同实体或实体群组相关的描述列表。
请将所有这些描述整合为一个单一且全面的描述，确保包含所有描述中的信息。
如果提供的描述有矛盾，请解决矛盾，并提供一个连贯的总结。
请确保使用第三人称撰写，并包含实体名称，以便我们有完整的上下文。
请不要加入提供的数据以外的内容。
请不要编造内容。
请不要加入不存在的实体内容。

#######
-数据-
实体: {entity_name}
描述列表: {description_list}
#######
输出:
"""

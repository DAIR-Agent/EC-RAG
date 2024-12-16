# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""实用提示：用于低级LLM调用。"""

JSON_CHECK_PROMPT = """
你将获得一个格式不正确的JSON字符串，它在执行json.loads时引发了错误。
它可能包含不必要的转义序列，或者某处缺少逗号或冒号。
你的任务是修复这个字符串并返回一个包含单个对象的格式正确的JSON字符串。
消除任何不必要的转义序列。
只返回可用json.loads解析的有效JSON字符串，这里指修复后的字符串本身而不是修复使用的代码，不需要额外说明。

# Examples
-----------
text: { \\"title\\": \\"甲乙丙\\", \\"summary\\": \\"定义\\" }
output: {"title": "甲乙丙", "summary": "定义"}
-----------
text: {"title": "甲乙丙", "summary": "定义"
output: {"title": "甲乙丙", "summary": "定义"}
-----------
text: {{"title': "甲乙丙", 'summary": "定义"
output: {"title": "甲乙丙", "summary": "定义"}
-----------
text: "{"title": "甲乙丙", "summary": "定义"}"
output: {"title": "甲乙丙", "summary": "定义"}
-----------
text: [{"title": "甲乙丙", "summary": "定义"}]
output: [{"title": "甲乙丙", "summary": "定义"}]
-----------
text: [{"title": "甲乙丙", "summary": "定义"}, { \\"title\\": \\"甲乙丙\\", \\"summary\\": \\"定义\\" }]
output: [{"title": "甲乙丙", "summary": "定义"}, {"title": "甲乙丙", "summary": "定义"}]
-----------
text: ```json\n[{"title": "甲乙丙", "summary": "定义"}, { \\"title\\": \\"甲乙丙\\", \\"summary\\": \\"定义\\" }]```
output: [{"title": "甲乙丙", "summary": "定义"}, {"title": "甲乙丙", "summary": "定义"}]


# Real Data
text: {input_text}
output:"""

"""Prompt template for checking if dialogue should end."""

DIALOGUE_END_CHECK_PROMPT = """请判断以下对话是否应该自然结束。

## 对话历史
{dialogue_history}

## 判断标准
对话应该结束的情况:
1. 用户的主要问题已经得到回答
2. 医生已经给出了明确的建议（如建议就医、建议观察等）
3. 用户表示理解或感谢
4. 对话自然收尾（如道别）
5. 对话陷入重复

对话应该继续的情况:
1. 用户还有未回答的问题
2. 医生还在询问关键信息
3. 讨论还在进行中
4. 用户表现出还想继续交流

## ⚠️ JSON 格式严格要求
1. 直接输出纯 JSON，不要有任何 markdown 代码块标记（不要 ```json）
2. 不要在 JSON 前后添加任何说明文字
3. should_end 必须是布尔值 true 或 false（不是字符串 "true" 或 "false"）
4. reason 必须是字符串

## 输出格式
{{
    "should_end": true,
    "reason": "判断理由字符串"
}}
"""

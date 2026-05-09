"""Prompt template for selecting starting event for dialogue."""

# Trap event types (consistent with trap_events.py)
TRAP_EVENT_TYPES_SET = {
    "allergy",
    "medication_history",
    "disease_history",
    "medication_preference",
    "diet_preference",
    "lifestyle_economic",
}

EVENT_SELECTION_PROMPT = """你是一个医疗对话数据集生成助手。请从以下事件列表中选择一个最适合作为健康咨询起点的事件。

## 用户画像
{persona_context}

## 可选事件列表
{events_list}

## 已选择的事件（避免重复）
{selected_events}

## 选择要求
1. **优先选择健康相关事件**（type=health），这些事件最适合作为医疗咨询的起点
2. 其他类型事件也可选择，如果它们可能引发健康问题（如工作压力导致失眠、家庭矛盾导致焦虑等）
3. 尽量避免选择已选事件列表中的事件
4. 如果必须复用事件（事件数量不足），请选择一个不同的对话角度：
   - 首诊：第一次咨询这个问题
   - 追问细节：针对某个具体症状深入询问
   - 寻求建议：症状有变化，想获得新的建议
   - 家属咨询：以家属身份替患者咨询

## ⚠️ JSON 格式严格要求
1. 直接输出纯 JSON，不要有任何 markdown 代码块标记（不要 ```json）
2. 不要在 JSON 前后添加任何说明文字
3. selected_event_id 必须是整数（如 1, 2, 3），不能是字符串
4. 其他字段必须是字符串

## 输出格式
{{
    "selected_event_id": 1,
    "event_summary": "事件简要描述字符串",
    "selection_reason": "选择该事件的理由字符串",
    "dialogue_angle": "首诊"
}}
"""


# Trap-priority event selection prompt (ensures all 6 trap types are covered)
EVENT_SELECTION_TRAP_PRIORITY_PROMPT = """你是一个医疗对话数据集生成助手。请从以下事件列表中选择一个最适合作为健康咨询起点的事件。

## 用户画像
{persona_context}

## 可选事件列表
{events_list}

## 已选择的事件（避免重复）
{selected_events}

## 尚未选择的陷阱事件类型（必须优先选择！）
{missing_trap_types}

## 选择要求（按优先级排序）

### ⚠️ 最高优先级：覆盖所有陷阱事件类型
如果「尚未选择的陷阱事件类型」不为空，**必须优先**从这些类型中选择一个事件！
陷阱事件类型说明：
- allergy: 过敏史相关事件
- medication_history: 用药史相关事件
- disease_history: 疾病史相关事件
- medication_preference: 给药偏好相关事件
- diet_preference: 饮食偏好相关事件
- lifestyle_economic: 生活&经济情况相关事件

### 次要优先级：选择健康相关事件
如果所有陷阱事件类型都已覆盖，则优先选择 health 类型的事件。

### 其他事件
其他类型事件（life, work）也可选择，如果它们可能引发健康问题。

## 对话角度
根据事件类型选择合适的对话角度：
- 陷阱事件：重点是如何在对话中**自然地透露**这些个人信息（过敏史、用药史等）
- health 事件：首诊、追问细节、寻求建议、复查反馈等
- 其他事件：压力咨询、健康担忧、预防咨询等

## ⚠️ JSON 格式严格要求
1. 直接输出纯 JSON，不要有任何 markdown 代码块标记
2. 不要在 JSON 前后添加任何说明文字
3. selected_event_id 必须是整数
4. 其他字段必须是字符串

## 输出格式
{{
    "selected_event_id": 1,
    "event_summary": "事件简要描述字符串",
    "selection_reason": "选择该事件的理由字符串",
    "dialogue_angle": "首诊/透露过敏史/提及用药史/等"
}}
"""

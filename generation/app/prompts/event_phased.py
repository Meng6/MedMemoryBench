"""Phased event generation prompt templates."""

# Phase name mapping
PHASE_NAMES = {
    1: "早期阶段",
    2: "病情转折阶段",
    3: "并发症演变阶段",
    4: "生活方式与心理挑战阶段",
    5: "复诊与好转阶段",
}

# Phase descriptions (defaults when no clinical report is available)
PHASE_DESCRIPTIONS = {
    1: "初始症状出现、首次就诊、初步检查与诊断",
    2: "病情变化、治疗方案调整、关键医疗决策",
    3: "并发症出现与演变、多系统影响、复杂治疗方案",
    4: "生活方式调整、心理压力应对、社会支持",
    5: "病情稳定、康复进展、长期健康管理规划",
}

# Suggested event type distribution per phase
PHASE_EVENT_TYPE_HINTS = {
    1: """
本阶段应重点关注：
- health 类型为主：症状出现、就诊、检查、初步诊断
- life 类型：因症状引起的生活困扰
- work 类型：工作受到影响""",

    2: """
本阶段应重点关注：
- health 类型为主：治疗效果评估、方案调整、新症状、复查
- life 类型：饮食运动调整的执行与挑战
- work 类型：病假、工作调整""",

    3: """
本阶段应重点关注：
- health 类型为主：并发症检查、专科会诊、联合治疗
- life 类型：生活方式的重大调整""",

    4: """
本阶段应重点关注：
- life 类型增加：饮食习惯、运动计划、作息调整
- health 类型：定期复查、指标监测
- work 类型：重返工作、职业调整""",

    5: """
本阶段应重点关注：
- health 类型为主：复查结果好转、药物减量、长期随访
- life 类型：健康生活方式的坚持与成效
- work 类型：工作与健康的平衡""",
}


# Main phased event generation prompt
EVENT_PHASED_GENERATE_PROMPT = """你是一个医疗健康领域的事件模拟专家。请基于用户画像和诊疗流程指导，生成该患者在当前阶段可能发生的健康事件。

## 用户画像
{persona_context}

## 当前生成阶段：{phase_name}（第 {phase_number}/5 阶段）

## ⚠️ 核心要求：严格遵循诊疗流程指导
以下是针对该用户疾病的**专业诊疗流程指导**，这是基于深度研究生成的权威报告。
你生成的事件**必须严格体现**报告中提到的：
1. **具体检查指标和数值**（如 HbA1c 具体百分比、血糖 mmol/L 数值、C肽水平等）
2. **具体药物名称和剂量**（如二甲双胍 1500mg/d、德谷胰岛素 14U 等）
3. **诊断过程和疾病演变**（如误诊、确诊、方案调整等关键节点）
4. **并发症和症状描述**（按报告中描述的时间线和严重程度）
5. **生活方式和心理状态**（报告中提到的具体场景和挑战）

### 诊疗流程指导原文（必须逐条对照生成事件）：
---
{phase_guidance}
---

## 已有事件时间线（包含固定性陷阱事件和已生成的常规事件）
{existing_events}

## ⚠️ 重要：关注已有的固定性陷阱事件
在「已有事件时间线」中，包含了用户的6类固定性陷阱事件（过敏史、用药史、疾病史、给药偏好、饮食偏好、生活经济情况）。
生成新事件时，**必须考虑这些陷阱事件对诊疗过程的影响**：
- 如果用户有药物过敏史，相关就诊/用药事件应体现医生避开该药物
- 如果用户有既往病史，治疗方案应考虑既往病史的影响
- 如果用户有给药偏好（如吞咽困难），处方应体现剂型调整
- 如果用户有饮食偏好，饮食指导事件应体现个性化调整
- 如果用户有经济限制，用药方案应体现经济性考量

## 生成要求
- 时间范围: {start_date} 到 {end_date}
- 本阶段生成 {num_events} 个事件
- 新事件的 temp_id 从 {start_temp_id} 开始递增
- 新事件可以引用已有事件的 ID（1 到 {max_existing_id}）作为 triggered_by
- 日期应合理分布在本阶段的时间范围内，保持时间顺序

## ⚠️ 必须遵循的生成原则（按优先级排序）
1. **【最高优先级】报告内容必须体现**：诊疗流程指导中提到的每一个关键事件（检查、诊断、用药、指标变化）都必须在生成的事件中有对应体现。不能自行编造与报告不符的内容。
2. **数值必须精确**：报告中提到的具体数值（HbA1c 9.2%、血糖 12.0 mmol/L 等）必须原样使用，不能随意修改。
3. **药物必须准确**：报告中提到的药物名称和剂量必须原样使用。
4. **时间线必须对应**：报告描述的是某个时间段的情况，生成的事件时间必须在对应的日期范围内。
5. **因果关系必须合理**：事件之间要有合理的因果关系和时间顺序。

## 阶段事件类型分布建议
{event_type_hints}

## 事件类型定义
- `health`: 健康相关（症状、就诊、检查、诊断、治疗、用药、复查、康复等）
- `life`: 生活调整（饮食、运动、作息、生活方式改变等）
- `work`: 工作影响（请假、工作调整、职业规划等）

## 事件描述规范
每个事件应包含 2-3 句话：
1. **核心事件**：具体发生了什么（必须与诊疗指导对应）
2. **专业细节**：检查结果数值、药物剂量、症状程度等（必须使用报告中的原始数据）
3. **影响/后续**：对生活的影响或下一步计划

好的示例（严格对应诊疗指导）：
✓ "首次确诊后开始口服降糖药治疗，起始方案为二甲双胍1500mg/d联合DPP-4抑制剂。用药1个月后复查HbA1c降至8.1%，空腹血糖有所改善。"
✓ "用药3个月后HbA1c意外反弹至8.8%，尽管严格遵医嘱用药和控制饮食，这种继发性失效引起医生对诊断的重新审视。"
✓ "紧急转诊至三甲医院，GADA抗体检测呈强阳性（>2000 U/mL），空腹C肽193 pmol/L，确诊为SAID而非最初判断的2型糖尿病。"

不好的示例：
✗ "去医院复查了。"（太简略，没有具体内容）
✗ "HbA1c 7.5%"（与报告数据不符，报告中是9.2%→8.1%→8.8%）
✗ "开始使用胰岛素"（没有体现报告中的具体方案：德谷胰岛素14U等）

## ⚠️ JSON 格式严格要求
1. 直接输出纯 JSON，不要有任何 markdown 代码块标记
2. 不要在 JSON 前后添加任何说明文字
3. temp_id 必须是整数，从 {start_temp_id} 开始
4. type 必须是: "health", "life", "work" 之一
5. event_date 必须是 "YYYY-MM-DD" 格式
6. triggered_by 必须是整数数组

## 输出格式
{{
    "events": [
        {{
            "temp_id": {start_temp_id},
            "event": "严格对应诊疗指导的详细事件描述，包含具体数值和药物",
            "type": "health",
            "event_date": "YYYY-MM-DD",
            "triggered_by": []
        }}
    ]
}}
"""


def build_phased_event_prompt(
    persona_context: str,
    phase_number: int,
    phase_guidance: str,
    existing_events: str,
    start_date: str,
    end_date: str,
    num_events: int,
    start_temp_id: int,
    max_existing_id: int,
) -> str:
    """Build the phased event generation prompt."""
    phase_name = PHASE_NAMES.get(phase_number, f"阶段{phase_number}")
    event_type_hints = PHASE_EVENT_TYPE_HINTS.get(phase_number, "")

    if not phase_guidance or phase_guidance.strip() == "":
        phase_guidance = f"本阶段重点：{PHASE_DESCRIPTIONS.get(phase_number, '疾病管理')}"

    return EVENT_PHASED_GENERATE_PROMPT.format(
        persona_context=persona_context,
        phase_name=phase_name,
        phase_number=phase_number,
        phase_guidance=phase_guidance,
        existing_events=existing_events,
        start_date=start_date,
        end_date=end_date,
        num_events=num_events,
        start_temp_id=start_temp_id,
        max_existing_id=max_existing_id,
        event_type_hints=event_type_hints,
    )

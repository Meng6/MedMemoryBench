"""Prompt template for persona enrichment.

Optimized version with Chain-of-Thought reasoning, self-validation,
and structured output format for consistent quality.

Key constraints:
- Use absolute time (e.g., "August 2024") instead of relative time
- Use fictional company/institution names
- Do NOT generate conversation_style fields (removed to avoid evaluation leakage)
- STRICT JSON format compliance required
- Support disease progression phases from deep-research report
"""

import json
from pathlib import Path

# Cache for user_report data
_user_report_cache: dict | None = None


def _load_user_report() -> dict:
    """Load user_report.json."""
    global _user_report_cache
    if _user_report_cache is not None:
        return _user_report_cache

    report_path = Path(__file__).parent.parent.parent / "data" / "user_report.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            _user_report_cache = json.load(f)
    else:
        _user_report_cache = {}
    return _user_report_cache


def _get_report_for_persona(persona_id: int) -> dict | None:
    """Get report data for a specific persona."""
    reports = _load_user_report()
    for report_key, report_data in reports.items():
        if report_data.get("persona_id") == persona_id:
            return report_data
    return None


PERSONA_ENRICH_PROMPT = """你是一位资深的医疗健康用户研究专家，擅长构建真实、立体的用户画像。

## 任务
基于原始画像信息，生成丰富且内在一致的扩展画像。

## 原始画像
- ID: {id}
- 类型: {type_name}
- 性别: {gender}
- 核心特征: {core_feature}
- 健康目标: {health_goals}
- 分类: {category}

## 生成流程

### 第一步：画像理解
<thinking>
分析此用户的核心特征，推断：
1. 最可能的年龄段和职业背景
2. 与核心特征匹配的生活方式
3. 导致当前健康状况的可能原因
4. 此类用户的典型使用场景
</thinking>

### 第二步：一致性构建
基于第一步的分析，确保以下逻辑链条成立：
- 职业 → 生活方式 → 健康状况
- 年龄 → 关注重点

### 第三步：生成扩展字段
按以下结构生成，每个字段都应与核心特征呼应：

**基础信息**
- age_range: 字符串，年龄范围（如 "35-40岁"）
- occupation_detail: 字符串，职业详情（用"某XX公司/机构"替代真实名称）

**lifestyle（生活方式）** - 必须是对象
- sleep_pattern: 字符串，睡眠模式
- diet_habits: 字符串，饮食习惯
- exercise_frequency: 字符串，运动频率
- stress_level: 字符串，压力水平及来源

**health_details（健康细节）** - 必须是对象
- medical_history: 字符串数组，病史（不要包含具体年份，使用相对描述如"初期"、"确诊后"等）

**background_story（背景故事）**
- 字符串，150-250字，第三人称叙述
- 整合以上信息，呈现完整的健康管理情境
- 不要包含具体年份，使用相对时间描述

### 第四步：自我检查
<validation>
逐项检查：
□ 无具体年份时间表达（如 2024年）？使用相对描述即可
□ 无真实公司/机构名称？
□ 所有数组字段都是数组类型（用 [] 包裹）？
□ 所有对象字段都是对象类型（用 {{}} 包裹）？
□ 各字段之间无逻辑矛盾？
□ 未生成任何对话风格相关字段？
</validation>

## ⚠️ JSON 格式严格要求
1. 直接输出纯 JSON，不要有任何 markdown 代码块标记（不要 ```json）
2. 不要在 JSON 前后添加任何说明文字
3. 确保 JSON 语法正确（正确的引号、逗号、括号匹配）
4. 所有字符串值使用双引号
5. 数组用 []，对象用 {{}}

## 输出格式
{{
    "age_range": "年龄范围字符串",
    "occupation_detail": "职业详情字符串",
    "lifestyle": {{
        "sleep_pattern": "睡眠模式字符串",
        "diet_habits": "饮食习惯字符串",
        "exercise_frequency": "运动频率字符串",
        "stress_level": "压力水平字符串"
    }},
    "health_details": {{
        "medical_history": ["病史1", "病史2"]
    }},
    "background_story": "背景故事字符串"
}}"""


PERSONA_ENRICH_PROMPT_WITH_REPORT = """你是一位资深的医疗健康用户研究专家，擅长构建真实、立体的用户画像。

## 任务
基于原始画像信息和疾病进展报告，生成丰富且内在一致的扩展画像。

## 原始画像
- ID: {id}
- 类型: {type_name}
- 性别: {gender}
- 核心特征: {core_feature}
- 健康目标: {health_goals}
- 分类: {category}

## ⚠️ 疾病进展报告（Deep-Research Report）
以下是该患者的详细疾病进展报告，你需要基于此报告生成画像。

**注意：报告中的具体年份/月份仅供参考疾病发展时间线，在生成画像时请转换为相对时间描述（如"初期"、"确诊后3个月"等），不要在输出中包含具体年份。**

### 第一阶段：误诊与代偿期
{phase_1}

### 第二阶段：病情转折点
{phase_2}

### 第三阶段：并发症演变
{phase_3}

### 第四阶段：生活方式与心理挑战
{phase_4}

### 第五阶段：复诊与疾病好转
{phase_5}

## 生成流程

### 第一步：报告理解
<thinking>
分析疾病进展报告，理解：
1. 患者的核心疾病及其特殊性（如SAID vs T2DM）
2. 疾病发展的各个阶段和关键转折点
3. 误诊期的症状表现和患者认知
4. 确诊后的治疗方案变化
5. 患者面临的生活和心理挑战
</thinking>

### 第二步：画像与报告对齐
确保生成的画像与报告中的疾病进展逻辑一致：
- 职业特点与报告中描述的工作环境一致
- 生活方式与报告中描述的习惯一致
- 病史描述要体现报告中的疾病阶段特点

### 第三步：生成扩展字段

**基础信息**
- age_range: 字符串，年龄范围（如 "28-32岁"）
- occupation_detail: 字符串，职业详情（用"某XX公司/机构"替代真实名称）

**lifestyle（生活方式）** - 必须是对象
- sleep_pattern: 字符串，睡眠模式（参考报告中的描述）
- diet_habits: 字符串，饮食习惯（参考报告中的描述）
- exercise_frequency: 字符串，运动频率
- stress_level: 字符串，压力水平及来源（参考报告中的描述）

**health_details（健康细节）** - 必须是对象
- medical_history: 字符串数组，病史
  - **重要**：不要包含具体年份！使用相对时间描述
  - **重要**：病史要体现疾病进展的阶段性，但不要提前泄露后期信息
  - 示例：["初期出现间歇性视力模糊和口渴症状", "曾被误诊为2型糖尿病", "后经详细检查确诊为自身免疫性糖尿病"]
- disease_progression: 对象，疾病进展阶段摘要
  - phase_1: 字符串，第一阶段（误诊与代偿期）的关键信息摘要
  - phase_2: 字符串，第二阶段（病情转折点）的关键信息摘要
  - phase_3: 字符串，第三阶段（并发症演变）的关键信息摘要
  - phase_4: 字符串，第四阶段（生活方式与心理挑战）的关键信息摘要
  - phase_5: 字符串，第五阶段（复诊与疾病好转）的关键信息摘要

**background_story（背景故事）**
- 字符串，150-250字，第三人称叙述
- 整合以上信息，呈现完整的健康管理情境
- **不要包含具体年份**，使用相对时间描述
- **不要提前泄露疾病的真实诊断**（如在故事中直接说"患有SAID"）

### 第四步：自我检查
<validation>
逐项检查：
□ 无具体年份时间表达（如 2024年、去年8月）？
□ 无真实公司/机构名称？
□ 所有数组字段都是数组类型（用 [] 包裹）？
□ 所有对象字段都是对象类型（用 {{}} 包裹）？
□ disease_progression 包含所有五个阶段？
□ 各字段之间无逻辑矛盾？
□ 背景故事没有提前泄露疾病的真实诊断？
</validation>

## ⚠️ JSON 格式严格要求
1. 直接输出纯 JSON，不要有任何 markdown 代码块标记（不要 ```json）
2. 不要在 JSON 前后添加任何说明文字
3. 确保 JSON 语法正确（正确的引号、逗号、括号匹配）
4. 所有字符串值使用双引号
5. 数组用 []，对象用 {{}}

## 输出格式
{{
    "age_range": "年龄范围字符串",
    "occupation_detail": "职业详情字符串",
    "lifestyle": {{
        "sleep_pattern": "睡眠模式字符串",
        "diet_habits": "饮食习惯字符串",
        "exercise_frequency": "运动频率字符串",
        "stress_level": "压力水平字符串"
    }},
    "health_details": {{
        "medical_history": ["病史1", "病史2", "病史3"],
        "disease_progression": {{
            "phase_1": "第一阶段摘要",
            "phase_2": "第二阶段摘要",
            "phase_3": "第三阶段摘要",
            "phase_4": "第四阶段摘要",
            "phase_5": "第五阶段摘要"
        }}
    }},
    "background_story": "背景故事字符串"
}}"""


def build_enrich_prompt(persona: dict) -> str:
    """Build the enrichment prompt for a given persona.

    Args:
        persona: Base persona dict from user_personas.json

    Returns:
        Formatted prompt string
    """
    persona_id = persona["id"]
    report = _get_report_for_persona(persona_id)

    if report and report.get("phase_1"):
        return PERSONA_ENRICH_PROMPT_WITH_REPORT.format(
            id=persona["id"],
            type_name=persona["type_name"],
            gender=persona["gender"],
            core_feature=persona["core_feature"],
            health_goals="、".join(persona["health_goals"]),
            category=persona["category"],
            phase_1=report.get("phase_1", ""),
            phase_2=report.get("phase_2", ""),
            phase_3=report.get("phase_3", ""),
            phase_4=report.get("phase_4", ""),
            phase_5=report.get("phase_5", ""),
        )
    else:
        return PERSONA_ENRICH_PROMPT.format(
            id=persona["id"],
            type_name=persona["type_name"],
            gender=persona["gender"],
            core_feature=persona["core_feature"],
            health_goals="、".join(persona["health_goals"]),
            category=persona["category"],
        )

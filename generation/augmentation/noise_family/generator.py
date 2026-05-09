"""Family/friends noise session generator.

Generates family/friend roles for user personas and produces continuous
health consultation dialogues for each role. Supports inter-session
continuity via key-point summaries for context passing.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from .config import FamilyNoiseConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Prompts
# ============================================================================

ROLE_GENERATION_PROMPT = """【重要声明】本任务是医学人工智能学术研究项目的一部分，用于构建医疗对话基准数据集（MedMemoryBench）。生成的所有内容均为**完全虚构的角色设定**，仅用于：
1. 训练和评估医疗AI助手的对话能力
2. 测试AI系统对复杂健康咨询场景的理解
3. 学术论文发表和科研用途

所有生成内容不会用于实际医疗决策，请放心生成符合要求的虚构角色数据。

---

请根据用户画像信息，生成该用户身边的亲人朋友角色。每个角色必须有完整、详细的健康档案，能够支撑长期、深入的健康咨询。

## 用户画像
{persona_info}

## 生成要求

### 1. 基本信息（每个角色）
生成 {num_roles} 个角色，每个包含：
- relationship: 与用户的关系（父亲/母亲/配偶/公婆/朋友/孩子等）
- name: 口语化称呼（如"我爸"、"老张"、"小李"）
- age_range: 具体年龄范围（如"62-65岁"）
- occupation: 职业或身份
- personality: 性格特点（影响就医态度和依从性）
- lifestyle_habits: 生活习惯概述（饮食、运动、作息、烟酒等）

### 2. 健康档案（每个角色3-5个健康问题，这是重点！）
每个 health_condition 必须包含以下详细信息：

**基础信息：**
- condition_name: 疾病/症状的准确医学名称
- icd_category: 疾病大类（如心血管、内分泌、骨科、呼吸系统等）
- severity: 严重程度（轻度/中度/重度）+ 具体说明
- duration: 确切患病时长

**病情详情：**
- diagnosis_history: 诊断经过（什么时候、怎么发现的、做过什么检查）
- symptoms_detail: 具体症状描述（频率、程度、诱因、缓解因素）
- recent_changes: 近期变化（最近1-2周的症状变化）
- lab_results: 最近的检查/化验结果（要有具体数值）

**治疗情况：**
- medications: 详细用药方案，格式：药名+剂量+频次+用药时长+效果评价
- treatment_history: 既往治疗史（做过什么治疗、效果如何）
- doctor_recommendations: 医生曾给出的建议

**关注点：**
- concerns: 家属最担心的问题（具体化）
- questions_to_ask: 家属想咨询的具体问题列表（3-5个）
- upcoming_events: 即将发生的医疗事件（复查、手术、换药等）

### 3. 角色多样化要求
- 老年人（父母/祖父母）：慢性病管理（高血压、糖尿病、冠心病、骨关节病、慢阻肺等），常有多病共存
- 中年人（配偶/兄弟姐妹）：亚健康、职业病、压力相关疾病（颈椎病、脂肪肝、胃病、焦虑抑郁）
- 儿童/青少年（孩子）：生长发育、免疫力、过敏、近视、心理健康
- 朋友同事：可以是特殊病种（肿瘤康复、自身免疫病、罕见病等）

### 4. 健康问题关联性
同一角色的健康问题应有内在联系，例如：
- 糖尿病 → 糖尿病视网膜病变风险 → 糖尿病肾病监测
- 高血压 → 冠心病 → 高血脂
- 长期久坐 → 颈椎病 + 腰椎间盘突出 + 脂肪肝

## 输出格式
返回JSON数组，确保每个角色的健康档案足够详细、真实、可信，能够支撑至少20轮深入的医疗咨询对话。

```json
[
  {{
    "relationship": "父亲",
    "name": "我爸",
    "age_range": "62-65岁",
    "occupation": "退休工人",
    "personality": "性格倔强，怕麻烦，不爱去医院",
    "lifestyle_habits": "饮食偏咸，不爱运动，有30年烟龄（每天1包），偶尔喝酒",
    "health_conditions": [
      {{
        "condition_name": "2型糖尿病",
        "icd_category": "内分泌代谢",
        "severity": "中度 - 血糖控制不稳定",
        "duration": "确诊5年",
        "diagnosis_history": "2019年单位体检发现空腹血糖升高(7.8mmol/L)，后到医院做糖耐量试验确诊",
        "symptoms_detail": "偶尔口渴多饮，视物模糊时有发生，双脚偶有麻木感",
        "recent_changes": "最近一周血糖波动较大，餐后血糖经常超过12mmol/L",
        "lab_results": "最近一次（2周前）：空腹血糖8.2mmol/L，糖化血红蛋白7.8%，尿微量白蛋白30mg/L",
        "medications": "二甲双胍缓释片0.5g bid（早晚餐后），服用3年，效果一般；格列美脲2mg qd（早餐前），新加1个月",
        "treatment_history": "最初单用二甲双胍，控制尚可。去年开始血糖升高，医生建议联合用药",
        "doctor_recommendations": "控制饮食，监测血糖，3个月复查糖化",
        "concerns": "担心发展成糖尿病并发症，特别是眼睛和肾脏",
        "questions_to_ask": ["血糖波动大是不是药不够", "脚麻是不是并发症的信号", "要不要打胰岛素", "饮食具体怎么控制"],
        "upcoming_events": "下个月预约眼底检查"
      }}
    ]
  }}
]
```

只返回JSON数组，不要其他内容。"""

FAMILY_USER_SYSTEM_PROMPT = """你是一个正在向在线医生咨询亲人健康问题的用户。你对亲人的病情非常了解，能说出详细的症状、用药和检查情况。

## 你的身份
{user_identity}

## 你要咨询的亲人信息
- 关系：{family_relationship}
- 称呼：{family_name}
- 年龄：{family_age}
- 职业：{family_occupation}
- 性格特点：{family_personality}

## 该亲人的完整健康档案
{health_conditions_detail}

## 本次咨询重点
{current_consultation_focus}

## 过往咨询记录摘要
{past_consultations}

## 对话要求

### 1. 提问专业度
- 能准确描述症状的部位、性质、频率、持续时间、诱因
- 能说出具体的检查结果数值（如"空腹血糖8.2"、"血压150/95"）
- 能说出完整的用药方案（药名、剂量、服用时间）
- 能描述用药后的效果和不良反应

### 2. 咨询连贯性
- 如果是首次咨询：完整介绍病情背景，提出最关心的问题
- 如果是后续咨询：
  * 先反馈上次医生建议的执行情况和效果
  * 描述病情的最新变化
  * 基于新情况提出进一步问题

### 3. 对话风格
- 像真正关心亲人的家属一样说话，体现焦虑和关心
- 可以表达困惑（"我不太理解..."、"这个指标是不是..."）
- 可以追问细节（"那具体应该怎么做"、"这个药能和XX一起吃吗"）
- 每次回复2-4句话，可以包含多个相关问题
- 适当使用口语化表达，但医学信息要准确

### 4. 围绕咨询重点展开
必须紧扣本次咨询重点 "{current_consultation_focus}" 来提问，但可以自然地关联到其他健康问题（因为慢性病往往相互影响）。

请直接以用户身份回复，不要添加任何角色标签或前缀。"""

FAMILY_DOCTOR_SYSTEM_PROMPT = """你是一位资深的在线问诊医生，有丰富的临床经验。你正在回复一位家属关于其亲人健康问题的咨询。

## 患者基本信息
- 与咨询者关系：咨询者的{family_relationship}
- 称呼：{family_name}
- 年龄：{family_age}

## 患者完整健康档案
{health_conditions_detail}

## 本次咨询重点
{current_consultation_focus}

## 回复要求

### 1. 专业性
- 回复要体现对患者整体病情的了解，不是泛泛而谈
- 能结合患者的具体检查结果、用药情况给出针对性建议
- 解释医学概念时要通俗易懂，但术语要准确
- 必要时说明建议背后的医学原理

### 2. 回复内容可以包括
- **病情解释**：解释症状/指标的含义、可能的原因
- **用药指导**：具体的药物调整建议（剂量、时间、注意事项）
- **生活方式**：饮食、运动、作息的具体建议（要可操作）
- **监测建议**：需要关注的指标、监测频率、记录方法
- **复查安排**：什么时候复查、查什么项目
- **警示信号**：需要警惕的危险症状，什么情况需要立即就医
- **追问细节**：如信息不足，可询问更多细节以给出准确建议

### 3. 安全原则
- 对于严重情况，明确建议线下就医，不要只给在线建议
- 不随意更改用药方案，重大调整建议咨询主治医生
- 提醒用药禁忌和相互作用

### 4. 回复风格
- 语气专业但亲切，像一位负责任的医生
- 回复3-6句话，内容充实但不冗长
- 可以使用分点说明，但不要过度格式化
- 关键数值和注意事项要明确

请直接以医生身份回复，不要添加任何角色标签或前缀。"""

CONSULTATION_FOCUS_PROMPT = """根据亲人的健康档案和过往咨询历史，规划本次咨询的具体重点话题。

## 亲人健康档案
{health_conditions_detail}

## 过往咨询历史
{past_consultations}

## 咨询重点规划策略

### 如果是首次咨询：
选择以下之一作为切入点：
1. 最紧迫的问题（近期有明显加重或变化的症状）
2. 最困扰日常生活的问题
3. 即将进行的医疗事件（复查、手术等）相关准备
4. 家属最担心的并发症风险

### 如果是后续咨询（有咨询历史）：
按优先级选择：
1. 上次咨询后的执行反馈 + 新出现的情况
2. 上次未完全解答的问题的深入追问
3. 病情的新变化、新症状
4. 复查结果解读或下一步治疗方案
5. 之前提到要关注的事项的跟进

### 话题具体化要求：
话题必须具体、明确，包含：
- 具体的症状/指标/药物名称
- 时间节点或变化描述
- 明确的咨询目的

**好的示例：**
- "父亲最近一周餐后血糖经常超过12，想问是否需要调整用药方案"
- "上次医生建议的二甲双胍加量后，胃不舒服，想问有什么替代方案"
- "下周要做眼底检查，想提前了解检查注意事项和可能的结果"
- "母亲骨密度又下降了，钙片换成什么品牌比较好"

**不好的示例：**
- "血糖问题"（太宽泛）
- "糖尿病咨询"（没有具体点）
- "身体不舒服"（不明确）

## 输出
请直接返回一个具体的咨询话题（一句话，15-40字），不要其他任何内容。"""

EXTRACT_SUMMARY_PROMPT = """请从以下对话中提取关键信息摘要，用于后续咨询时的上下文参考。

## 亲人信息
- 关系：{family_relationship}
- 称呼：{family_name}

## 本次咨询重点
{consultation_focus}

## 对话内容
{dialogue_text}

## 要求
提取3-5个关键要点，包括：
1. 本次咨询的主要问题
2. 医生给出的具体建议和指导
3. 需要后续关注或复查的事项
4. 用药或生活方式的调整建议
5. 下一步行动计划（如有）

请用简洁的中文描述，每个要点一行，用"-"开头。只返回摘要内容，不要其他说明。
"""


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class HealthCondition:
    """Health condition/disease with detailed health profile."""

    # Basic info
    condition_name: str  # Exact medical name of the disease/symptom
    icd_category: str = ""  # Disease category (cardiovascular, endocrine, etc.)
    severity: str = ""  # Severity level with details
    duration: str = ""  # Exact duration of illness

    # Condition details
    diagnosis_history: str = ""  # Diagnosis history
    symptoms_detail: str = ""  # Detailed symptom description
    recent_changes: str = ""  # Recent changes
    lab_results: str = ""  # Recent lab/test results

    # Treatment info
    medications: str = ""  # Detailed medication plan
    treatment_history: str = ""  # Past treatment history
    doctor_recommendations: str = ""  # Previous doctor recommendations

    # Concerns
    concerns: str = ""  # Family member's primary concerns
    questions_to_ask: List[str] = field(default_factory=list)  # Specific questions to consult
    upcoming_events: str = ""  # Upcoming medical events

    # Legacy field
    current_status: str = ""  # Current status (backward compatible)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "condition_name": self.condition_name,
            "icd_category": self.icd_category,
            "severity": self.severity,
            "duration": self.duration,
            "diagnosis_history": self.diagnosis_history,
            "symptoms_detail": self.symptoms_detail,
            "recent_changes": self.recent_changes,
            "lab_results": self.lab_results,
            "medications": self.medications,
            "treatment_history": self.treatment_history,
            "doctor_recommendations": self.doctor_recommendations,
            "concerns": self.concerns,
            "questions_to_ask": self.questions_to_ask,
            "upcoming_events": self.upcoming_events,
            "current_status": self.current_status,
        }

    def to_summary(self) -> str:
        """Convert to brief summary text."""
        parts = [f"【{self.condition_name}】"]
        if self.severity:
            parts.append(f"{self.severity}")
        if self.duration:
            parts.append(f"，病程{self.duration}")
        if self.current_status:
            parts.append(f"，{self.current_status}")
        if self.medications:
            parts.append(f"。用药：{self.medications}")
        if self.concerns:
            parts.append(f"。家属担心：{self.concerns}")
        return "".join(parts)

    def to_detail(self) -> str:
        """Convert to detailed health profile text."""
        lines = [f"### {self.condition_name}"]

        # Basic info
        basic_info = []
        if self.icd_category:
            basic_info.append(f"分类：{self.icd_category}")
        if self.severity:
            basic_info.append(f"程度：{self.severity}")
        if self.duration:
            basic_info.append(f"病程：{self.duration}")
        if basic_info:
            lines.append(" | ".join(basic_info))

        if self.diagnosis_history:
            lines.append(f"**诊断经过**：{self.diagnosis_history}")

        if self.symptoms_detail:
            lines.append(f"**症状表现**：{self.symptoms_detail}")

        if self.recent_changes:
            lines.append(f"**近期变化**：{self.recent_changes}")

        if self.lab_results:
            lines.append(f"**检查Result**：{self.lab_results}")

        if self.medications:
            lines.append(f"**用药方案**：{self.medications}")

        if self.treatment_history:
            lines.append(f"**治疗史**：{self.treatment_history}")

        if self.doctor_recommendations:
            lines.append(f"**医生建议**：{self.doctor_recommendations}")

        if self.concerns:
            lines.append(f"**家属担心**：{self.concerns}")

        if self.questions_to_ask:
            lines.append(f"**想咨询的Question**：" + "；".join(self.questions_to_ask))

        if self.upcoming_events:
            lines.append(f"**近期安排**：{self.upcoming_events}")

        return "\n".join(lines)


@dataclass
class FamilyRole:
    """Family/friend role."""

    role_id: int  # Role ID (unique within the persona)
    persona_id: int  # Parent user persona ID
    relationship: str  # Relationship to the user
    name: str  # Informal name/title
    age_range: str  # Age range
    occupation: str  # Occupation
    personality: str  # Personality traits
    health_conditions: List[HealthCondition]  # List of health conditions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role_id": self.role_id,
            "persona_id": self.persona_id,
            "relationship": self.relationship,
            "name": self.name,
            "age_range": self.age_range,
            "occupation": self.occupation,
            "personality": self.personality,
            "health_conditions": [hc.to_dict() for hc in self.health_conditions],
        }

    def get_health_conditions_detail(self) -> str:
        """Get detailed health profile text."""
        if not self.health_conditions:
            return "暂无详细健康记录"

        lines = []
        for i, hc in enumerate(self.health_conditions, 1):
            lines.append(f"## 健康Question {i}")
            lines.append(hc.to_detail())
            lines.append("")  # blank line separator

        return "\n".join(lines)


@dataclass
class FamilyNoiseMessage:
    """Family consultation session message."""

    turn: int
    role: str  # "user" or "assistant"
    content: str
    agent_type: str  # "user_agent" or "doctor_agent"


@dataclass
class FamilyNoiseSession:
    """Family/friends noise session."""

    noise_family_id: int  # Noise identifier
    noise_type: str  # "family_health_consultation"
    persona_id: int  # Parent user persona ID
    family_role: FamilyRole  # Family role info
    health_issue: str  # Health issue for this consultation
    turn_count: int
    messages: List[FamilyNoiseMessage]
    knowledge_points: List[Dict[str, Any]]
    session_summary: str  # Session summary (context for subsequent sessions)
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        # Process messages: may be FamilyNoiseMessage objects or dicts
        messages_list = []
        for msg in self.messages:
            if isinstance(msg, FamilyNoiseMessage):
                messages_list.append({
                    "turn": msg.turn,
                    "role": msg.role,
                    "content": msg.content,
                    "agent_type": msg.agent_type,
                })
            else:
                messages_list.append(msg)
        
        return {
            "noise_family_id": self.noise_family_id,
            "noise_type": self.noise_type,
            "persona_id": self.persona_id,
            "family_role": self.family_role.to_dict(),
            "health_issue": self.health_issue,
            "turn_count": self.turn_count,
            "messages": messages_list,
            "knowledge_points": self.knowledge_points,
            "session_summary": self.session_summary,
            "created_at": self.created_at,
        }


# ============================================================================
# Generator
# ============================================================================

class FamilyDialogueGenerator:
    """Family/friends noise session generator.

    Generates family roles for user personas and produces continuous
    health consultation dialogues for each role.
    """

    def __init__(self, config: Optional[FamilyNoiseConfig] = None):
        """Initialize generator.

        Args:
            config: Configuration object.
        """
        self.config = config or FamilyNoiseConfig()
        self._load_api_config()
        self._client = None

        # Role and session records
        self._roles_by_persona: Dict[int, List[FamilyRole]] = {}
        self._role_past_summaries: Dict[str, List[str]] = {}  # role_key -> summaries

        # Per-role knowledge points (role_key -> knowledge_points)
        self._role_knowledge_points: Dict[str, List[Dict[str, Any]]] = {}

        logger.info("[FamilyDialogueGenerator] Initializecomplete")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  每个角色的Session数: {self.config.sessions_per_role}")

    def _load_api_config(self) -> None:
        """Load API config."""
        import os
        from dotenv import load_dotenv

        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)  # Force override existing env vars

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
        model = self.config.model or os.getenv("LLM_MODEL") or os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
        self.model = model.replace("openai/", "") if model.startswith("openai/") else model

    @property
    def client(self):
        """Get OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def _get_completion_kwargs(self, max_tokens: int) -> Dict[str, Any]:
        """Get completion parameters compatible with different models."""
        legacy_model_prefixes = (
            "gpt-3.5",
            "gpt-4-",
            "text-davinci",
            "text-curie",
            "text-babbage",
            "text-ada",
        )

        if any(self.model.startswith(prefix) for prefix in legacy_model_prefixes):
            return {"max_tokens": max_tokens}
        else:
            return {"max_completion_tokens": max_tokens}

    def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        caller: str = "unknown",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Unified LLM call method with token tracking.

        Args:
            messages: LLM message list.
            caller: Caller identifier for tracking.
            temperature: Temperature parameter (defaults to config value).
            max_tokens: Max token count (defaults to config value).

        Returns:
            LLM response content.
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from app.services.token_tracker import get_token_tracker

        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
            **self._get_completion_kwargs(tokens),
        )

        # Track token usage
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            tracker = get_token_tracker()
            tracker.set_model(self.model)
            tracker.track(
                prompt_tokens=usage.prompt_tokens or 0,
                completion_tokens=usage.completion_tokens or 0,
                total_tokens=usage.total_tokens or 0,
                caller=f"noise_family.{caller}",
            )

        content = response.choices[0].message.content
        return content.strip() if content else ""

    def load_personas(self) -> List[Dict[str, Any]]:
        """Load user persona data.

        Returns:
            List of user personas.
        """
        personas_path = Path(self.config.data_dir) / self.config.personas_filename
        logger.info(f"[FamilyDialogueGenerator] LoadUser persona: {personas_path}")

        with open(personas_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        personas = data.get("personas", [])
        logger.info(f"  Load了 {len(personas)} 个User persona")
        return personas

    async def generate_family_roles(self, persona: Dict[str, Any]) -> List[FamilyRole]:
        """Generate family/friend roles for a single user persona.

        Args:
            persona: User persona data.

        Returns:
            List of generated family roles with detailed health conditions.
        """
        persona_id = persona["persona_id"]
        logger.info(f"[FamilyDialogueGenerator] 为 Persona {persona_id} Generatefamily角色")

        # Build user persona info
        base_info = persona.get("base_info", {})
        enriched = persona.get("enriched_data", {})

        persona_info = f"""
- 疾病类型: {base_info.get('type_name', 'N/A')}
- 性别: {base_info.get('gender', 'N/A')}
- 年龄范围: {enriched.get('age_range', 'N/A')}
- 职业: {enriched.get('occupation_detail', 'N/A')}
- 背景故事: {enriched.get('background_story', 'N/A')[:300]}...
"""

        prompt = ROLE_GENERATION_PROMPT.format(
            persona_info=persona_info,
            num_roles=self.config.num_family_roles,
        )

        try:
            content = self._call_llm(
                [{"role": "user", "content": prompt}],
                caller="generate_family_roles",
                max_tokens=8000,
            )

            # Check if response content is empty
            if not content:
                logger.warning("[FamilyDialogueGenerator] LLM Return内容为空，使用Default角色")
                default_roles = self._generate_default_roles(persona_id)
                self._roles_by_persona[persona_id] = default_roles
                return default_roles

            logger.debug(f"[FamilyDialogueGenerator] LLM 原始Return（前500字符）: {content[:500]}")

            # Parse JSON - try multiple extraction methods
            json_content = content
            if "```json" in content:
                parts = content.split("```json")
                if len(parts) > 1:
                    json_content = parts[1].split("```")[0].strip()
            elif "```" in content:
                parts = content.split("```")
                if len(parts) > 1:
                    json_content = parts[1].split("```")[0].strip()

            # If extracted content is empty, try parsing original content directly
            if not json_content:
                logger.warning("[FamilyDialogueGenerator] JSON 提取Result为空，尝试直接ParseOriginal content")
                json_content = content

            # Try to find the start and end of JSON array
            if json_content and not json_content.startswith("["):
                start_idx = json_content.find("[")
                if start_idx != -1:
                    end_idx = json_content.rfind("]")
                    if end_idx != -1:
                        json_content = json_content[start_idx:end_idx + 1]

            logger.debug(f"[FamilyDialogueGenerator] 提取的 JSON（前500字符）: {json_content[:500] if json_content else '空'}")

            roles_data = json.loads(json_content)

            # Convert to FamilyRole objects
            roles = []
            for idx, role_data in enumerate(roles_data):
                # Parse health conditions list (supports detailed fields)
                health_conditions = []
                for hc_data in role_data.get("health_conditions", []):
                    # Process questions_to_ask field
                    questions = hc_data.get("questions_to_ask", [])
                    if isinstance(questions, str):
                        questions = [q.strip() for q in questions.split("；") if q.strip()]

                    hc = HealthCondition(
                        condition_name=hc_data.get("condition_name", ""),
                        icd_category=hc_data.get("icd_category", ""),
                        severity=hc_data.get("severity", "中度"),
                        duration=hc_data.get("duration", ""),
                        diagnosis_history=hc_data.get("diagnosis_history", ""),
                        symptoms_detail=hc_data.get("symptoms_detail", ""),
                        recent_changes=hc_data.get("recent_changes", ""),
                        lab_results=hc_data.get("lab_results", ""),
                        medications=hc_data.get("medications", "无"),
                        treatment_history=hc_data.get("treatment_history", ""),
                        doctor_recommendations=hc_data.get("doctor_recommendations", ""),
                        concerns=hc_data.get("concerns", ""),
                        questions_to_ask=questions,
                        upcoming_events=hc_data.get("upcoming_events", ""),
                        current_status=hc_data.get("current_status", ""),
                    )
                    health_conditions.append(hc)

                role = FamilyRole(
                    role_id=idx + 1,
                    persona_id=persona_id,
                    relationship=role_data.get("relationship", "family"),
                    name=role_data.get("name", "family"),
                    age_range=role_data.get("age_range", "未知"),
                    occupation=role_data.get("occupation", "未知"),
                    personality=role_data.get("personality", ""),
                    health_conditions=health_conditions,
                )
                roles.append(role)

                logger.info(f"  角色 {idx+1}: {role.name}({role.relationship}), "
                           f"{len(health_conditions)} 个健康Question")

            self._roles_by_persona[persona_id] = roles
            return roles

        except json.JSONDecodeError as e:
            logger.error(f"[FamilyDialogueGenerator] JSON ParseFailed: {e}")
            logger.error(f"[FamilyDialogueGenerator] 待Parse内容（前1000字符）: {json_content[:1000] if 'json_content' in dir() and json_content else '空'}")
            logger.error(f"[FamilyDialogueGenerator] 原始Return（前1000字符）: {content[:1000] if 'content' in dir() and content else '空'}")
            # Return default roles
            default_roles = self._generate_default_roles(persona_id)
            self._roles_by_persona[persona_id] = default_roles
            return default_roles
        except Exception as e:
            logger.error(f"[FamilyDialogueGenerator] 角色GenerateFailed: {e}")
            # Return default roles
            default_roles = self._generate_default_roles(persona_id)
            self._roles_by_persona[persona_id] = default_roles
            return default_roles

    def _generate_default_roles(self, persona_id: int) -> List[FamilyRole]:
        """Generate default family roles with detailed health conditions."""
        default_configs = [
            {
                "relationship": "父亲",
                "name": "我爸",
                "age_range": "62-65岁",
                "occupation": "退休工人",
                "personality": "性格倔强，怕麻烦，不爱去医院",
                "health_conditions": [
                    HealthCondition(
                        condition_name="2型糖尿病",
                        icd_category="内分泌代谢",
                        severity="中度 - 血糖控制不稳定",
                        duration="确诊5年",
                        diagnosis_history="2019年单位体检发现空腹血糖升高(7.8mmol/L)，后到医院做糖耐量试验确诊",
                        symptoms_detail="偶尔口渴多饮，视物模糊时有发生，双脚偶有麻木感",
                        recent_changes="最近一周血糖波动较大，餐后血糖经常超过12mmol/L",
                        lab_results="最近一次（2周前）：空腹血糖8.2mmol/L，糖化血红蛋白7.8%，尿微量白蛋白30mg/L",
                        medications="二甲双胍缓释片0.5g bid（早晚餐后），服用3年；格列美脲2mg qd（早餐前），新加1个月",
                        treatment_history="最初单用二甲双胍控制尚可，去年Start血糖升高后联合用药",
                        doctor_recommendations="控制饮食，监测血糖，3个月复查糖化血红蛋白",
                        concerns="担心发展成糖尿病并发症，特别是眼睛和肾脏",
                        questions_to_ask=["血糖波动大是不是药不够", "脚麻是不是并发症的信号", "要不要打胰岛素"],
                        upcoming_events="下个月预约眼底检查",
                    ),
                    HealthCondition(
                        condition_name="高血压",
                        icd_category="心血管",
                        severity="中度 - 2级高血压",
                        duration="确诊8年",
                        diagnosis_history="2016年头晕就诊发现血压160/100mmHg，确诊高血压",
                        symptoms_detail="平时无明显症状，劳累或情绪激动时偶有头晕",
                        recent_changes="最近天气变化，血压有所波动，早上偶尔超过150/95",
                        lab_results="上个月复查：血压142/88mmHg，心电图大致正常，肾功能正常",
                        medications="苯磺酸氨氯地平5mg qd（晨起），服用5年，控制尚可",
                        treatment_history="最初用厄贝沙坦效果不佳，后换成氨氯地平控制较好",
                        doctor_recommendations="低盐饮食，每日监测血压，保持情绪稳定",
                        concerns="担心长期用药副作用，担心发展为心脏病或脑血管病",
                        questions_to_ask=["血压药能长期吃吗", "有没有副作用小的药", "血压多少算控制好"],
                        upcoming_events="",
                    ),
                    HealthCondition(
                        condition_name="腰椎间盘突出",
                        icd_category="骨科",
                        severity="中度 - L4/L5突出",
                        duration="确诊5年，反复发作",
                        diagnosis_history="2019年腰痛伴右腿放射痛就诊，CT显示L4/L5椎间盘突出",
                        symptoms_detail="腰部酸痛，久坐后加重，弯腰时右腿有牵扯感",
                        recent_changes="最近帮忙搬东西后又Start疼，右腿麻木感明显",
                        lab_results="去年MRI：L4/L5椎间盘突出，硬膜囊受压",
                        medications="发作时用双氯芬酸钠止痛贴，疼痛剧烈时口服塞来昔布",
                        treatment_history="做过3个疗程理疗，效果一般。医生建议过手术，但不想做",
                        doctor_recommendations="避免久坐弯腰，加强腰背肌锻炼，严重时考虑手术",
                        concerns="怕手术风险，但又担心病情加重会瘫痪",
                        questions_to_ask=["不做手术能好吗", "有什么保守治疗方法", "日常怎么锻炼"],
                        upcoming_events="",
                    ),
                ],
            },
            {
                "relationship": "母亲",
                "name": "我妈",
                "age_range": "60-63岁",
                "occupation": "退休教师",
                "personality": "爱操心，容易焦虑，对健康Question比较敏感",
                "health_conditions": [
                    HealthCondition(
                        condition_name="骨质疏松症",
                        icd_category="骨科/内分泌",
                        severity="中度 - T值-2.8",
                        duration="确诊4年",
                        diagnosis_history="2020年体检骨密度检查发现，当时腰椎T值-2.5",
                        symptoms_detail="偶有腰背部隐痛，身高较年轻时矮了2cm",
                        recent_changes="上次复查骨密度又下降了，T值从-2.5降到-2.8",
                        lab_results="3个月前骨密度：腰椎T值-2.8，股骨颈T值-2.3。血钙正常，维生素D偏低(18ng/ml)",
                        medications="钙尔奇D 600mg 每日一片，阿法骨化醇0.25μg每日一次",
                        treatment_history="一直在补钙，但效果不明显，医生建议加用双膦酸盐类药物",
                        doctor_recommendations="继续补钙补D，考虑抗骨质疏松药物，注意防跌倒",
                        concerns="非常担心骨折，特别怕髋部骨折后卧床",
                        questions_to_ask=["骨密度还能恢复吗", "双膦酸盐有什么副作用", "日常怎么预防骨折"],
                        upcoming_events="下周预约内分泌科门诊",
                    ),
                    HealthCondition(
                        condition_name="失眠症",
                        icd_category="神经/心理",
                        severity="轻到中度",
                        duration="2年多",
                        diagnosis_history="退休后Start出现睡眠Question，入睡困难，经常凌晨3点醒来就睡不着",
                        symptoms_detail="入睡need1-2小时，半夜易醒，早醒后难以再入睡，白天疲乏",
                        recent_changes="最近因为担心骨质疏松的事，睡眠更差了",
                        lab_results="未做专门检查",
                        medications="阿普唑仑0.4mg 睡前（仅在实在睡不着时吃），每周吃2-3次",
                        treatment_history="试过褪黑素、酸枣仁丸等，效果不明显",
                        doctor_recommendations="规律作息，睡前不要看手机，必要时药物助眠",
                        concerns="担心安眠药上瘾，但不吃又睡不着",
                        questions_to_ask=["怎么能不吃药也睡好", "安眠药能长期吃吗", "有没有中药调理的方法"],
                        upcoming_events="",
                    ),
                    HealthCondition(
                        condition_name="甲状腺结节",
                        icd_category="内分泌",
                        severity="轻度 - TI-RADS 3类",
                        duration="发现1年",
                        diagnosis_history="去年体检B超发现甲状腺左侧叶一个0.8cm结节，边界清，无钙化",
                        symptoms_detail="无明显症状，不疼不痒，吞咽无异常",
                        recent_changes="半年前复查结节无明显变化，仍为0.8cm",
                        lab_results="甲功正常：TSH 2.1mIU/L，FT3、FT4均正常。B超：左叶结节0.8×0.6cm，TI-RADS 3类",
                        medications="无",
                        treatment_history="医生建议定期随访观察",
                        doctor_recommendations="每6个月复查B超，观察结节变化",
                        concerns="非常担心会癌变，看到网上说的甲状腺癌就害怕",
                        questions_to_ask=["结节会不会癌变", "需不need做穿刺", "平时饮食要注意什么"],
                        upcoming_events="2个月后复查B超",
                    ),
                ],
            },
            {
                "relationship": "配偶",
                "name": "老公",
                "age_range": "32-35岁",
                "occupation": "程序员",
                "personality": "工作狂，经常加班，不太注意身体，讳疾忌医",
                "health_conditions": [
                    HealthCondition(
                        condition_name="颈椎病",
                        icd_category="骨科",
                        severity="轻度 - 颈型",
                        duration="2年多",
                        diagnosis_history="长期对着电脑工作，颈部酸痛，去年拍片显示颈椎生理曲度变直",
                        symptoms_detail="颈部僵硬酸痛，低头工作2小时以上加重，偶有向肩部放射",
                        recent_changes="最近项目忙加班多，颈椎疼痛加重，有时还头晕",
                        lab_results="颈椎X光：生理曲度变直，C5-C6椎间隙稍窄。未做MRI",
                        medications="疼痛时贴膏药，偶尔按摩",
                        treatment_history="买过颈椎按摩仪，用了几次就搁置了",
                        doctor_recommendations="纠正坐姿，每小时活动颈部，建议做颈椎操",
                        concerns="担心发展成颈椎间盘突出，影响工作",
                        questions_to_ask=["颈椎病会越来越严重吗", "有什么好的治疗方法", "need做MRI吗"],
                        upcoming_events="",
                    ),
                    HealthCondition(
                        condition_name="脂肪肝",
                        icd_category="消化内科",
                        severity="轻度",
                        duration="发现1年",
                        diagnosis_history="去年公司体检B超发现，当时转氨酶正常",
                        symptoms_detail="无明显症状，偶有右上腹不适感",
                        recent_changes="最近体检转氨酶有点高了，ALT 68U/L",
                        lab_results="B超：肝脏回声增强，提示轻度脂肪肝。ALT 68U/L(正常<40)，AST 45U/L",
                        medications="无",
                        treatment_history="医生说要减肥运动，但一直没行动",
                        doctor_recommendations="控制体重，低脂饮食，加强运动，戒酒",
                        concerns="担心发展成肝硬化或肝癌",
                        questions_to_ask=["转氨酶高要不要吃药", "脂肪肝能逆转吗", "多久复查一次"],
                        upcoming_events="",
                    ),
                    HealthCondition(
                        condition_name="慢性胃炎",
                        icd_category="消化内科",
                        severity="轻度 - 浅表性",
                        duration="3年",
                        diagnosis_history="饮食不规律，经常胃痛胃胀，做胃镜显示慢性浅表性胃炎，幽门螺杆菌阴性",
                        symptoms_detail="进食后胃胀，空腹时胃部隐痛，嗳气反酸",
                        recent_changes="最近加班吃外卖多，胃不舒服次数增加",
                        lab_results="2年前胃镜：慢性浅表性胃炎。HP(-)。",
                        medications="胃不舒服时吃奥美拉唑20mg和铝碳酸镁",
                        treatment_history="吃过一段Time胃药好转，但停药后反复",
                        doctor_recommendations="规律饮食，少吃刺激性食物，避免空腹太久",
                        concerns="担心发展成胃溃疡或更严重的Question",
                        questions_to_ask=["need再做胃镜吗", "怎么才能根治", "长期吃奥美拉唑有副作用吗"],
                        upcoming_events="",
                    ),
                ],
            },
            {
                "relationship": "friend",
                "name": "老王",
                "age_range": "38-42岁",
                "occupation": "销售总监",
                "personality": "应酬多，喜欢喝酒，生活不规律，对健康比较大意",
                "health_conditions": [
                    HealthCondition(
                        condition_name="痛风/高尿酸血症",
                        icd_category="内分泌代谢",
                        severity="中度 - 有过急性发作",
                        duration="确诊2年",
                        diagnosis_history="2年前夜间右脚大脚趾剧痛红肿，急诊查尿酸580μmol/L，确诊痛风",
                        symptoms_detail="急性期过后无明显症状，但尿酸一直偏高",
                        recent_changes="最近应酬多，上周又有点脚趾隐痛，担心又要发作",
                        lab_results="上个月：尿酸495μmol/L(正常<420)，肝肾功能正常",
                        medications="非布司他40mg每日一次，服用1年多",
                        treatment_history="急性发作用秋水仙碱+消炎痛控制，之后长期服用降尿酸药",
                        doctor_recommendations="严格忌口，禁酒，多喝水，坚持服药",
                        concerns="担心再次发作，那个疼实在受不了",
                        questions_to_ask=["吃了药尿酸还高正常吗", "能不能偶尔喝点酒", "这个药要吃一辈子吗"],
                        upcoming_events="",
                    ),
                    HealthCondition(
                        condition_name="高血脂",
                        icd_category="心血管/代谢",
                        severity="轻度 - 以甘油三酯升高为主",
                        duration="发现1年多",
                        diagnosis_history="体检发现，与长期应酬饮酒有关",
                        symptoms_detail="无明显症状",
                        recent_changes="甘油三酯从2.8升到3.2mmol/L",
                        lab_results="总胆固醇5.8mmol/L，甘油三酯3.2mmol/L(正常<1.7)，低密度脂蛋白3.5mmol/L",
                        medications="暂未用药",
                        treatment_history="医生建议先生活方式干预3个月",
                        doctor_recommendations="戒酒，低脂饮食，增加运动",
                        concerns="担心发展成动脉硬化、冠心病",
                        questions_to_ask=["需不need吃降脂药", "高血脂有什么症状", "甘油三酯高和胆固醇高哪个更危险"],
                        upcoming_events="下月复查血脂",
                    ),
                    HealthCondition(
                        condition_name="焦虑Status",
                        icd_category="心理/神经",
                        severity="轻度",
                        duration="半年左右",
                        diagnosis_history="工作压力大，业绩考核重，出现紧张、心慌、失眠，未正式就诊",
                        symptoms_detail="工作时容易紧张，开会前心慌手抖，晚上睡眠浅，易惊醒",
                        recent_changes="最近业绩压力大，症状有加重",
                        lab_results="未做检查",
                        medications="无，不想吃药",
                        treatment_history="自己买过一些安神的保健品，效果不明显",
                        doctor_recommendations="未就诊",
                        concerns="不想被认为是心理有Question，也担心吃药会影响工作",
                        questions_to_ask=["这算是焦虑症吗", "不吃药能好吗", "有什么调节的方法"],
                        upcoming_events="",
                    ),
                ],
            },
            {
                "relationship": "孩子",
                "name": "宝宝",
                "age_range": "4-5岁",
                "occupation": "幼儿园小friend",
                "personality": "活泼好动，有点挑食，对打针吃药比较抗拒",
                "health_conditions": [
                    HealthCondition(
                        condition_name="反复呼吸道感染",
                        icd_category="儿科/呼吸",
                        severity="轻度 - 每年感冒6-8次",
                        duration="1年多",
                        diagnosis_history="上幼儿园后Start频繁生病，换季时尤其明显",
                        symptoms_detail="感冒症状为主：流涕、咳嗽、发热，每次持续5-7天",
                        recent_changes="这个月已经感冒2次了，刚好又Start流鼻涕",
                        lab_results="上次感冒查血常规：白细胞正常，淋巴细胞比例偏高",
                        medications="感冒时对症用药：小儿氨酚黄那敏、易坦静等",
                        treatment_history="吃过一段Time的维生素和益生菌，效果不明显",
                        doctor_recommendations="加强营养，增加户外活动，流感季节注意防护",
                        concerns="担心是不是免疫力太低，会不会影响生长发育",
                        questions_to_ask=["need查免疫功能吗", "怎么提高免疫力", "要不要打流感疫苗"],
                        upcoming_events="准备预约儿童保健科检查",
                    ),
                    HealthCondition(
                        condition_name="过敏性鼻炎",
                        icd_category="儿科/耳鼻喉",
                        severity="轻度 - 间歇性",
                        duration="半年多",
                        diagnosis_history="早晨起床后连续打喷嚏、流清鼻涕，医生说是过敏性鼻炎",
                        symptoms_detail="早晨和接触灰尘后明显，打喷嚏、流清涕、揉鼻子揉眼睛",
                        recent_changes="最近换季症状加重，晚上睡觉有点鼻塞",
                        lab_results="查过过敏原：尘螨2级阳性，其他阴性",
                        medications="生理盐水洗鼻，症状重时用糠酸莫米松喷鼻",
                        treatment_history="洗鼻有一定效果，但孩子不太配合",
                        doctor_recommendations="坚持洗鼻，保持室内清洁，必要时用鼻喷激素",
                        concerns="担心发展成哮喘，听说过敏性鼻炎和哮喘有关系",
                        questions_to_ask=["会发展成哮喘吗", "鼻喷激素对孩子有影响吗", "能根治吗"],
                        upcoming_events="",
                    ),
                    HealthCondition(
                        condition_name="挑食/体重偏轻",
                        icd_category="儿科/营养",
                        severity="轻度 - 体重在第15百分位",
                        duration="1年多",
                        diagnosis_history="不爱吃蔬菜和肉，主食吃得也少，体检体重一直偏轻",
                        symptoms_detail="吃饭磨蹭，对新食物抗拒，喜欢吃零食，正餐吃几口就说饱了",
                        recent_changes="最近生病后胃口更差了",
                        lab_results="身高105cm（P50），体重15kg（P15）。血红蛋白110g/L（略偏低），微量元素铁、锌偏低",
                        medications="补充维生素AD、葡萄糖酸锌口服液",
                        treatment_history="试过各种方法，效果不持久",
                        doctor_recommendations="培养良好进食习惯，增加食物多样性，必要时补充营养素",
                        concerns="担心营养不良影响生长发育和智力发展",
                        questions_to_ask=["need吃什么营养品吗", "怎么让孩子爱吃饭", "会影响长高吗"],
                        upcoming_events="",
                    ),
                ],
            },
        ]

        roles = []
        for idx, cfg in enumerate(default_configs[:self.config.num_family_roles]):
            role = FamilyRole(
                role_id=idx + 1,
                persona_id=persona_id,
                relationship=cfg["relationship"],
                name=cfg["name"],
                age_range=cfg["age_range"],
                occupation=cfg["occupation"],
                personality=cfg["personality"],
                health_conditions=cfg["health_conditions"],
            )
            roles.append(role)

        return roles

    def _get_role_key(self, persona_id: int, role_id: int) -> str:
        """Get unique identifier key for a role."""
        return f"{persona_id}_{role_id}"

    def _get_past_consultations_for_role(self, persona_id: int, role_id: int) -> str:
        """Get past consultation records for a role."""
        role_key = self._get_role_key(persona_id, role_id)
        summaries = self._role_past_summaries.get(role_key, [])

        if not summaries:
            return "（这是首次为这位family咨询）"

        # Show recent consultation records
        recent = summaries[-5:]
        lines = [f"第{i+1}次咨询记录：" for i in range(len(recent))]
        result = []
        for i, summary in enumerate(recent):
            result.append(f"【第{i+1}次咨询】\n{summary}")

        return "\n\n".join(result)

    async def _select_consultation_focus(
        self,
        role: FamilyRole,
        session_idx: int,
    ) -> str:
        """Intelligently select the focus topic for this consultation.

        Args:
            role: Family role.
            session_idx: Session index (which consultation number).

        Returns:
            Focus topic for this consultation.
        """
        past_consultations = self._get_past_consultations_for_role(role.persona_id, role.role_id)
        health_conditions_detail = role.get_health_conditions_detail()

        prompt = CONSULTATION_FOCUS_PROMPT.format(
            health_conditions_detail=health_conditions_detail,
            past_consultations=past_consultations,
        )

        try:
            focus = self._call_llm(
                [{"role": "user", "content": prompt}],
                caller="_select_consultation_focus",
                max_tokens=500,
            )
            # Strip possible quotes
            focus = focus.strip('"\'')
            return focus
        except Exception as e:
            logger.warning(f"[FamilyDialogueGenerator] 咨询重点选择Failed: {e}")
            # Round-robin selection of health conditions
            if role.health_conditions:
                idx = session_idx % len(role.health_conditions)
                hc = role.health_conditions[idx]
                return f"关于{role.name}的{hc.condition_name}的情况咨询"
            return f"关于{role.name}的健康状况咨询"

    async def _generate_user_turn(
        self,
        persona: Dict[str, Any],
        role: FamilyRole,
        consultation_focus: str,
        messages: List[FamilyNoiseMessage],
        turn: int,
    ) -> str:
        """Generate user turn."""
        enriched = persona.get("enriched_data", {})
        user_identity = f"年龄 {enriched.get('age_range', '未知')}，职业 {enriched.get('occupation_detail', '未知')}"

        past_consultations = self._get_past_consultations_for_role(role.persona_id, role.role_id)

        system_prompt = FAMILY_USER_SYSTEM_PROMPT.format(
            user_identity=user_identity,
            family_relationship=role.relationship,
            family_name=role.name,
            family_age=role.age_range,
            family_occupation=role.occupation,
            family_personality=role.personality,
            health_conditions_detail=role.get_health_conditions_detail(),
            current_consultation_focus=consultation_focus,
            past_consultations=past_consultations,
        )

        llm_messages = [{"role": "system", "content": system_prompt}]

        # Add dialogue history
        for msg in messages:
            llm_role = "assistant" if msg.agent_type == "user_agent" else "user"
            llm_messages.append({"role": llm_role, "content": msg.content})

        # First turn prompt
        if turn == 1:
            llm_messages.append({
                "role": "user",
                "content": f"请根据family的健康档案和本次咨询重点，向医生提出你的第一个Question。"
            })

        try:
            return self._call_llm(llm_messages, caller="_generate_user_turn")
        except Exception as e:
            logger.error(f"[FamilyDialogueGenerator] User轮次GenerateFailed: {e}")
            return f"医生您好，我想咨询一下{role.name}的{consultation_focus}。"

    async def _generate_doctor_turn(
        self,
        role: FamilyRole,
        consultation_focus: str,
        messages: List[FamilyNoiseMessage],
    ) -> str:
        """Generate doctor turn."""
        system_prompt = FAMILY_DOCTOR_SYSTEM_PROMPT.format(
            family_relationship=role.relationship,
            family_name=role.name,
            family_age=role.age_range,
            health_conditions_detail=role.get_health_conditions_detail(),
            current_consultation_focus=consultation_focus,
        )

        llm_messages = [{"role": "system", "content": system_prompt}]

        for msg in messages:
            llm_role = "user" if msg.agent_type == "user_agent" else "assistant"
            llm_messages.append({"role": llm_role, "content": msg.content})

        try:
            return self._call_llm(llm_messages, caller="_generate_doctor_turn")
        except Exception as e:
            logger.error(f"[FamilyDialogueGenerator] 医生轮次GenerateFailed: {e}")
            return "根据您描述的情况，我给您几点建议..."

    async def _extract_session_summary(
        self,
        role: FamilyRole,
        consultation_focus: str,
        messages: List[FamilyNoiseMessage],
    ) -> str:
        """Extract session summary."""
        dialogue_text = "\n".join([
            f"{'User' if m.agent_type == 'user_agent' else '医生'}: {m.content}"
            for m in messages
        ])

        prompt = EXTRACT_SUMMARY_PROMPT.format(
            family_relationship=role.relationship,
            family_name=role.name,
            consultation_focus=consultation_focus,
            dialogue_text=dialogue_text,
        )

        try:
            return self._call_llm(
                [{"role": "user", "content": prompt}],
                caller="_extract_session_summary",
                max_tokens=800,
            )
        except Exception as e:
            logger.warning(f"[FamilyDialogueGenerator] 摘要提取Failed: {e}")
            return f"关于{role.name}的{consultation_focus}咨询"

    async def _extract_knowledge_points(
        self,
        role: FamilyRole,
        consultation_focus: str,
        messages: List[FamilyNoiseMessage],
        noise_family_id: int,
        past_knowledge_points: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract knowledge points from dialogue.

        Args:
            role: Family role.
            consultation_focus: Consultation focus topic.
            messages: Dialogue message list.
            noise_family_id: Noise ID.
            past_knowledge_points: Historical knowledge points for this role.

        Returns:
            List of knowledge points (1-3 items).
        """
        dialogue_text = "\n".join([
            f"{'User' if m.agent_type == 'user_agent' else '医生'}: {m.content}"
            for m in messages
        ])

        # Build historical knowledge points summary for this role
        past_kp_text = ""
        if past_knowledge_points:
            past_kp_summary = "\n".join([
                f"- [{kp.get('category', '')}] {kp.get('name', '')}: {kp.get('content', '')}"
                for kp in past_knowledge_points[-10:]
            ])
            past_kp_text = f"""
## 关于{role.name}的历史知识点记录（参考，避免重复）
{past_kp_summary}
"""

        prompt = f"""请从以下关于亲人健康咨询的对话中提取1-3个关键知识点。
{past_kp_text}
## 亲人信息
- 关系：{role.relationship}
- 称呼：{role.name}
- 年龄：{role.age_range}

## 本次咨询重点
{consultation_focus}

## 对话内容
{dialogue_text}

## 要求
1. 提取1-3个本次对话的核心知识点（必须至少提取1个）
2. 避免与该亲人的历史知识点重复
3. 每个知识点包含：
   - category: 分类（如"用药指导"、"生活调理"、"复查提醒"、"症状观察"等）
   - name: 知识点名称（2-8个字）
   - content: 具体内容摘要（一句话，要具体实用）

请以JSON格式返回数组，只返回JSON数组，不要其他内容。
"""

        try:
            content = self._call_llm(
                [{"role": "user", "content": prompt}],
                caller="_extract_knowledge_points",
                max_tokens=1000,
            )

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            kps = json.loads(content)

            if not isinstance(kps, list):
                kps = [kps]

            if len(kps) > 3:
                kps = kps[:3]

            for kp in kps:
                kp["trap_score"] = 0.1
                kp["noise_family_id"] = noise_family_id
                kp["is_family_noise"] = True
                kp["family_role"] = role.name

            if len(kps) == 0:
                kps = [{
                    "category": "family健康",
                    "name": f"{role.name}咨询",
                    "content": f"关于{role.name}的{consultation_focus}",
                    "trap_score": 0.1,
                    "noise_family_id": noise_family_id,
                    "is_family_noise": True,
                    "family_role": role.name,
                }]

            return kps
        except Exception as e:
            logger.warning(f"[FamilyDialogueGenerator] Knowledge points提取Failed: {e}")
            return [{
                "category": "family健康",
                "name": f"{role.name}咨询",
                "content": f"关于{role.name}的{consultation_focus}",
                "trap_score": 0.1,
                "noise_family_id": noise_family_id,
                "is_family_noise": True,
                "family_role": role.name,
            }]

    async def generate_session(
        self,
        noise_family_id: int,
        persona: Dict[str, Any],
        role: FamilyRole,
        session_idx: int,
    ) -> FamilyNoiseSession:
        """Generate a single family consultation noise session.

        Args:
            noise_family_id: Noise ID.
            persona: User persona.
            role: Family role.
            session_idx: Session index for this role (for continuity).

        Returns:
            Generated noise session.
        """
        # Select consultation focus
        consultation_focus = await self._select_consultation_focus(role, session_idx)

        logger.info(f"[FamilyDialogueGenerator] GenerateSession {noise_family_id}: "
                   f"{role.name}({role.relationship}) - {consultation_focus}")

        # Determine dialogue turn count
        num_turns = random.randint(self.config.min_turns, self.config.max_turns)

        messages = []
        for turn in range(1, num_turns + 1):
            # User turn
            user_content = await self._generate_user_turn(
                persona, role, consultation_focus, messages, turn
            )
            messages.append(FamilyNoiseMessage(
                turn=turn,
                role="user",
                content=user_content,
                agent_type="user_agent",
            ))

            # Doctor turn
            doctor_content = await self._generate_doctor_turn(
                role, consultation_focus, messages
            )
            messages.append(FamilyNoiseMessage(
                turn=turn,
                role="assistant",
                content=doctor_content,
                agent_type="doctor_agent",
            ))

        # Get historical knowledge points for this role
        role_key = self._get_role_key(role.persona_id, role.role_id)
        role_past_kps = self._role_knowledge_points.get(role_key, [])

        # Extract session summary and knowledge points
        session_summary = await self._extract_session_summary(role, consultation_focus, messages)
        knowledge_points = await self._extract_knowledge_points(
            role, consultation_focus, messages, noise_family_id, role_past_kps
        )

        # Add newly extracted knowledge points to the role's list
        if role_key not in self._role_knowledge_points:
            self._role_knowledge_points[role_key] = []
        self._role_knowledge_points[role_key].extend(knowledge_points)

        # Record summary for subsequent sessions
        if role_key not in self._role_past_summaries:
            self._role_past_summaries[role_key] = []
        self._role_past_summaries[role_key].append(session_summary)

        session = FamilyNoiseSession(
            noise_family_id=noise_family_id,
            noise_type="family_health_consultation",
            persona_id=role.persona_id,
            family_role=role,
            health_issue=consultation_focus,
            turn_count=num_turns,
            messages=messages,
            knowledge_points=knowledge_points,
            session_summary=session_summary,
            created_at=datetime.now().isoformat(),
        )

        if self.config.verbose:
            logger.info(f"  complete: {num_turns} 轮Dialogue, {len(knowledge_points)} 个Knowledge points")
            logger.info(f"  角色 {role.name} Knowledge points累计: {len(self._role_knowledge_points[role_key])} 个")

        return session

    async def generate_all_for_persona(
        self,
        persona: Dict[str, Any],
        start_id: int,
    ) -> List[FamilyNoiseSession]:
        """Generate all family consultation sessions for a single user persona.

        Args:
            persona: User persona.
            start_id: Starting noise ID.

        Returns:
            List of generated noise sessions.
        """
        persona_id = persona["persona_id"]
        logger.info(f"[FamilyDialogueGenerator] Start为 Persona {persona_id} Generatefamily咨询Session")

        # Generate family roles first (with detailed health conditions)
        roles = await self.generate_family_roles(persona)

        sessions = []
        current_id = start_id

        # Generate sessions for each role
        for role in roles:
            logger.info(f"  Process角色: {role.name}({role.relationship}), "
                       f"{len(role.health_conditions)} 个健康Question")
            for session_idx in range(self.config.sessions_per_role):
                session = await self.generate_session(
                    noise_family_id=current_id,
                    persona=persona,
                    role=role,
                    session_idx=session_idx,
                )
                sessions.append(session)
                current_id += 1

        logger.info(f"  Persona {persona_id} complete，共 {len(sessions)} 个Session")
        return sessions

    async def generate_all(self) -> List[FamilyNoiseSession]:
        """Generate all family/friends noise sessions.

        Returns:
            List of generated noise sessions.
        """
        # Load user personas
        personas = self.load_personas()

        all_sessions = []
        current_id = 1

        for persona in personas:
            sessions = await self.generate_all_for_persona(persona, current_id)
            all_sessions.extend(sessions)
            current_id += len(sessions)

        logger.info(f"[FamilyDialogueGenerator] Generatecomplete，共 {len(all_sessions)} 个NoiseSession")
        return all_sessions

    def save_sessions(
        self,
        sessions: List[FamilyNoiseSession],
        output_path: str,
    ) -> None:
        """Save noise sessions to file."""
        data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_noise_sessions": len(sessions),
                "total_turns": sum(s.turn_count for s in sessions),
                "noise_type": "family_health_consultation",
            },
            "noise_sessions": [s.to_dict() for s in sessions],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"[FamilyDialogueGenerator] NoiseSession已Save到: {output_path}")

    def save_roles(self, output_path: str) -> None:
        """Save generated family roles to file."""
        all_roles = []
        for persona_id, roles in self._roles_by_persona.items():
            for role in roles:
                all_roles.append(role.to_dict())

        data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_roles": len(all_roles),
            },
            "family_roles": all_roles,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"[FamilyDialogueGenerator] family角色已Save到: {output_path}")

"""Family/friends noise data configuration."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class FamilyNoiseConfig:
    """Family/friends noise data configuration."""

    data_dir: str = "data"
    personas_filename: str = "generated_personas.json"
    input_filename: str = "generated_dialogues.json"
    output_filename: str = "generated_dialogues_with_family_noise.json"

    num_family_roles: int = 5
    sessions_per_role: int = 20
    min_turns: int = 5
    max_turns: int = 8

    model: Optional[str] = None
    temperature: float = 1.0
    max_tokens: int = 2000

    relationship_types: List[str] = field(default_factory=lambda: [
        "父亲",
        "母亲",
        "配偶",
        "子女",
        "兄弟姐妹",
        "祖父母",
        "叔叔/阿姨",
        "表兄弟姐妹",
        "亲密朋友",
        "同事",
    ])

    health_issue_categories: List[str] = field(default_factory=lambda: [
        "高血压日常管理",
        "糖尿病饮食控制",
        "冠心病用药咨询",
        "关节炎疼痛缓解",
        "慢性胃炎调理",
        "骨质疏松预防",
        "甲状腺功能异常",
        "慢性支气管炎护理",
        "记忆力下降担忧",
        "睡眠质量差",
        "便秘问题处理",
        "跌倒预防措施",
        "老年人营养补充",
        "听力下降应对",
        "腰椎间盘突出",
        "颈椎病防治",
        "脂肪肝调理",
        "更年期综合症",
        "压力相关症状",
        "近视度数加深",
        "胃食管反流",
        "偏头痛发作",
        "过敏性鼻炎",
        "皮肤问题咨询",
        "小儿发热处理",
        "儿童咳嗽护理",
        "小儿腹泻调理",
        "儿童营养发育",
        "儿童视力保护",
        "术后伤口护理",
        "康复期运动指导",
        "术后饮食注意",
        "复查时间安排",
        "焦虑情绪疏导",
        "抑郁倾向关注",
        "失眠焦虑调节",
        "情绪管理建议",
    ])

    verbose: bool = True
    dry_run: bool = False

"""Noise data augmentation configuration."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class NoiseConfig:
    """Noise data augmentation configuration."""

    data_dir: str = "data"
    input_filename: str = "generated_dialogues.json"
    output_filename: str = "generated_dialogues_with_noise.json"

    num_noise_sessions: int = 20
    min_turns: int = 3
    max_turns: int = 8

    model: Optional[str] = None
    temperature: float = 1.0
    max_tokens: int = 1500

    health_topics: List[str] = field(default_factory=lambda: [
        "早晨嗓子有异物感咳不出来",
        "持续头痛但不发烧",
        "经常感觉胸闷气短",
        "最近总是失眠多梦",
        "吃完饭后胃胀不消化",
        "皮肤突然起红疹很痒",
        "眼睛干涩看东西模糊",
        "腰酸背痛久坐不舒服",
        "手脚发麻是什么原因",
        "经常感觉疲劳没精神",
        "口腔溃疡反复发作",
        "耳鸣听力下降",
        "关节疼痛活动受限",
        "便秘或腹泻交替出现",
        "心跳加速心慌",
        "感冒什么情况需要打针",
        "发烧多少度需要吃退烧药",
        "抗生素什么时候该吃",
        "体检报告怎么看",
        "血压血糖正常值是多少",
        "疫苗接种的注意事项",
        "药物之间有什么禁忌",
        "中药西药能一起吃吗",
        "手术后多久能正常活动",
        "慢性病需要长期吃药吗",
        "如何预防季节性流感",
        "办公室久坐如何保护颈椎",
        "熬夜对身体有什么危害",
        "怎样提高免疫力",
        "饮食如何搭配更健康",
        "适合中老年人的运动方式",
        "如何保护眼睛预防近视",
        "睡眠质量差怎么改善",
        "压力大如何调节情绪",
        "如何预防三高",
        "孕期需要注意什么",
        "儿童发育期营养补充",
        "老年人骨质疏松预防",
        "女性生理期注意事项",
        "青少年心理健康问题",
        "产后恢复注意事项",
        "更年期综合症应对",
        "婴幼儿常见疾病护理",
        "减肥期间如何健康饮食",
        "戒烟戒酒有什么好方法",
        "运动损伤如何处理",
        "长期看手机对健康的影响",
        "保健品有必要吃吗",
        "中医养生的基本原则",
        "什么情况需要去医院检查",
        "家庭常备药有哪些",
    ])

    verbose: bool = True
    dry_run: bool = False

"""MCD (Multi-hop Clinical Deduction) Query generation prompts.

Three-phase generation architecture:
1. Phase 1 - Causal Chain Mining & Reasoning
2. Phase 2 - Chain Validation & Refinement
3. Phase 3 - Question & Answer Synthesis
"""

# ========== Phase 1: Causal Chain Mining & Medical Reasoning ==========

MCD_PHASE1_CAUSAL_CHAIN_MINING_PROMPT = """你是一位资深的内分泌科临床专家和医学推理专家。

## 核心任务
从患者的就诊时间线中挖掘具有**严密医学因果关系**的跨会话推理链。这些推理链需要展现专业的临床推理能力。

## ⚠️ 时间限制：当前 Session ID = {current_session_id}
**只能使用 Session ID ≤ {current_session_id} 的信息！**

## 患者事件时间线
{events_timeline}

## 患者知识点库（按 session 分组）
{knowledge_points_by_session}

## 已生成的推理链（避免重复！）
{existing_chains_hint}

---

## 推理链挖掘指南

### 1. 医学推理链的核心要素

一条高质量的推理链必须包含：
- **触发因素**：具体的药物、生活事件、检查指标变化
- **病理生理机制**：明确的医学机制解释（如药物作用、代谢通路）
- **临床表现**：患者可观察到的症状或指标变化
- **因果闭环**：从触发到结果的完整逻辑链条

### 2. 优先挖掘的推理模式

**模式A：药物-代谢相互作用**
```
具体药物 → 药理作用机制 → 代谢影响 → 血糖变化
例：布洛芬长期使用 → 抑制肾前列腺素合成 → GFR下降/胰岛素清除延迟 → 血糖波动
```

**模式B：应激-内分泌反应**
```
应激事件 → 神经内分泌反应 → 激素变化 → 血糖影响
例：连续熬夜 → 交感神经激活 → 皮质醇/肾上腺素升高 → 肝糖输出增加 → 空腹血糖升高
```

**模式C：器官功能-药物效应**
```
器官功能问题 → 药物代谢改变 → 治疗效果变化
例：脂肪肝 → 肝脏首过效应改变 → 口服降糖药生物利用度变化 → 血糖控制不稳
```

**模式D：饮食-血糖动力学**
```
特定饮食模式 → 胃肠吸收变化 → 血糖曲线改变
例：高GI碳水集中摄入 → 快速吸收 → 餐后血糖峰值过高 → 胰岛素分泌滞后
```

**模式E：病程进展-治疗失效**
```
疾病进展标志 → 病理机制 → 原方案失效
例：C肽进行性下降 → β细胞功能衰竭 → 口服药效果减弱 → 需要胰岛素
```

### 3. 推理链复杂度要求

- **跳数**：3-5跳（优先4跳，展现深度推理）
- **跨会话**：涉及至少2-3个不同session的事件
- **时间跨度**：优先选择>14天的长期关联
- **医学深度**：必须包含明确的病理生理机制节点

### 4. 节点类型定义

| 节点类型 | 说明 | 示例 |
|---------|------|------|
| **事实节点** | 来自对话/检查的客观信息 | "HbA1c 8.1%→8.8%反弹" |
| **机制节点** | 医学原理/病理生理解释 | "NSAIDs抑制COX-1导致肾血流减少" |
| **推理节点** | 基于事实的逻辑推断 | "胰岛素清除延迟+外源用量不变=低血糖风险" |

---

## ⚠️ 输出格式（纯JSON）

```json
{{
    "candidate_chains": [
        {{
            "chain_id": 1,
            "reasoning_pattern": "推理模式类型（A-E）",
            "core_mechanism": "核心病理生理机制（一句话概括）",
            "hop_count": 4,
            "nodes": [
                {{
                    "node_id": 1,
                    "node_type": "事实节点/机制节点/推理节点",
                    "session_id": 具体session_id或0（机制节点为0）,
                    "content": "节点内容（包含具体数值/药物名/时间）",
                    "role": "起始节点/中间节点/终点节点",
                    "source_info": "信息来源说明",
                    "medical_basis": "医学依据（仅机制节点需要）"
                }}
            ],
            "causal_explanation": "完整的因果关系医学解释（200-300字，专业详细）",
            "sessions_involved": [涉及的session_id列表],
            "quality_score": 0.85,
            "quality_reason": "评分理由"
        }}
    ],
    "mining_summary": {{
        "total_candidates": 数量,
        "patterns_found": ["发现的模式列表"],
        "best_candidate_id": 最佳候选ID,
        "selection_reason": "选择理由"
    }}
}}
```
"""

# ========== Phase 2: Chain Validation & Refinement ==========

MCD_PHASE2_CHAIN_VALIDATION_PROMPT = """你是一位严格的医学推理验证专家和临床内分泌学顾问。

## 核心任务
验证推理链的医学准确性，并精化节点内容，使其更加专业、精确、有临床价值。

## ⚠️ 时间限制：当前 Session ID = {current_session_id}
所有节点的 session_id 必须 ≤ {current_session_id}

## 待验证的推理链
{candidate_chain_json}

## 患者完整知识点库
{all_knowledge_points}

## 已生成的问题（避免重复）
{existing_queries_hint}

---

## 验证维度

### 1. 医学准确性验证（最重要！）

检查每个机制节点：
- 药理学机制是否正确？
- 病理生理解释是否符合当前医学共识？
- 因果关系方向是否正确？
- 是否有过度简化或错误推断？

**常见错误示例**：
- ❌ "布洛芬直接升高血糖" → 机制过度简化
- ✅ "布洛芬抑制肾前列腺素→入球小动脉收缩→GFR下降→胰岛素清除延迟"

### 2. 推理链完整性验证

- 从触发因素到最终结果是否有逻辑断层？
- 中间机制节点是否充分解释了因果转换？
- 是否遗漏了关键的中间环节？

### 3. 临床真实性验证

- 推理链中的现象在临床中是否常见？
- 时间关系是否符合实际病程发展？
- 效应量级是否合理？

### 4. 去重验证

- 与已生成问题的推理角度是否重复？
- 核心机制是否已被考察过？

---

## 精化要求

对每个节点进行精化：
1. **补充具体数值**：血糖值、HbA1c、药物剂量等
2. **明确时间信息**：具体日期、持续时间、时间间隔
3. **细化机制描述**：用专业术语准确描述
4. **增强因果连接**：明确说明"因为A所以B"的逻辑

---

## ⚠️ 输出格式（纯JSON）

```json
{{
    "validation_result": {{
        "is_valid": true/false,
        "overall_score": 0.85,
        "validation_details": {{
            "medical_accuracy": {{
                "passed": true/false,
                "score": 0.9,
                "issues": ["问题列表"],
                "corrections": ["修正建议"]
            }},
            "chain_completeness": {{
                "passed": true/false,
                "score": 0.8,
                "missing_links": ["缺失的环节"]
            }},
            "clinical_validity": {{
                "passed": true/false,
                "score": 0.85,
                "concerns": ["顾虑"]
            }},
            "uniqueness": {{
                "passed": true/false,
                "overlap_with": "如有重复，说明重复点"
            }}
        }}
    }},
    "refined_chain": {{
        "nodes": [
            {{
                "node_id": 1,
                "node_type": "事实节点/机制节点/推理节点",
                "session_id": session_id,
                "content": "精化后的内容（更专业、更具体）",
                "role": "起始节点/中间节点/终点节点",
                "source_info": "来源",
                "causal_link_to_next": "与下一节点的因果关系说明"
            }}
        ],
        "causal_explanation": "精化后的完整因果解释（专业详细，300-400字）",
        "required_memory_nodes": [
            "Session X: 必须记住的具体信息（含数值）"
        ],
        "core_mechanism_summary": "核心机制一句话总结"
    }},
    "rejection_reason": "如验证不通过，说明原因",
    "improvement_suggestions": ["改进建议"]
}}
```
"""

# ========== Phase 1.5: Chain Improvement (called after validation failure) ==========

MCD_PHASE1_5_CHAIN_IMPROVEMENT_PROMPT = """你是一位资深的医学推理专家。之前生成的推理链验证失败，请根据反馈改进。

## ⚠️ 时间限制：当前 Session ID = {current_session_id}

## 被拒绝的推理链
{rejected_chain_json}

## 拒绝原因
{rejection_reason}

## 改进建议
{improvement_suggestions}

## 患者事件时间线
{events_timeline}

## 患者知识点库
{knowledge_points_by_session}

## 已生成的推理链
{existing_chains_hint}

---

## 改进要求

1. **针对性修正**：直接解决被指出的问题
2. **保持医学严谨**：确保机制描述准确
3. **避免简单换皮**：如果原方向不可行，尝试全新角度
4. **满足复杂度**：3-5跳，跨至少2-3个session

---

## ⚠️ 输出格式（纯JSON）

```json
{{
    "improved_chain": {{
        "chain_id": 1,
        "reasoning_pattern": "推理模式",
        "core_mechanism": "核心机制",
        "hop_count": 4,
        "nodes": [节点列表，格式同阶段1],
        "causal_explanation": "因果解释",
        "sessions_involved": [session_id列表],
        "improvement_made": "相比之前做了哪些改进"
    }}
}}
```
"""

# ========== Phase 2.5: Content Enrichment (optional) ==========

MCD_PHASE2_5_CONTENT_ENRICHMENT_PROMPT = """你是一位临床内分泌专家。请基于真实对话内容进一步精化推理链。

## 当前 Session ID
{current_session_id}

## 待精化的推理链
{validated_chain_json}

## 相关 Session 的对话记录
{dialogues_content}

## 相关事件详情
{events_content}

## 知识点摘要
{kps_content}

---

## 精化任务

1. **提取具体数值**：从对话中找到精确的血糖值、HbA1c、药物剂量等
2. **补充时间细节**：具体日期、持续多久、间隔多长
3. **强化因果表述**：让每个节点间的因果关系更加清晰
4. **增加临床细节**：患者的具体症状描述、医生的具体建议等

---

## ⚠️ 输出格式（纯JSON）

```json
{{
    "enriched_chain": {{
        "nodes": [精化后的节点列表],
        "causal_explanation": "更详细的因果解释",
        "required_memory_nodes": ["需要记忆的关键信息"],
        "enrichment_summary": "精化了哪些内容"
    }}
}}
```
"""

# ========== Phase 3: Query Synthesis ==========

MCD_PHASE3_QUESTION_SYNTHESIS_PROMPT = """你是一位医疗对话数据集出题专家。请基于已验证的推理链，生成高质量的问答对。

## 核心原则

### 问题(Question)设计原则
1. **简洁自然**：用普通患者的口吻，不要像病例报告
2. **信息适量**：给出必要的背景，但不过度详细
3. **引导推理**：让问题自然指向需要推理的方向
4. **避免重复**：每个问题的切入点和表述方式要有差异

### 答案(Answer)设计原则
1. **医学专业**：使用准确的医学术语和机制解释
2. **逻辑清晰**：按推理链顺序展开，因果关系明确
3. **信息完整**：覆盖推理链的所有关键节点
4. **临床指导**：必要时给出合理的建议

---

## 当前 Session ID
{current_session_id}

## 问题序号
{query_idx}

## 已验证的推理链
{validated_chain_json}

## 患者背景
{background_summary}

## ⚠️ 已生成的问题（必须避免重复！）
{existing_questions_list}

---

## 问题设计指南

### ❌ 错误示例（问题太长太详细）
```
"我这两年因为肩颈一直不太舒服，医生给我开的止痛药我基本每天早上都会吃一粒。
今年年初那阵子加班特别厉害，肩颈又更紧了，我自己曾经有大概一两周早晚各吃一粒。
后来二月份回医院复诊时医生让我还是按原来的量吃。奇怪的是，今年一月份项目上线
那几天，我每天都忙到凌晨一两点才下班，中间还点了深夜外卖..."（太长！）
```

### ✅ 正确示例（简洁自然）

**类型1：症状+疑惑型**
```
"最近连着几天熬夜加班，早上血糖突然高了不少，明明饮食没变化，这是怎么回事？"
```

**类型2：观察+求因型**
```
"我发现只要前一天睡不好，第二天空腹血糖就会偏高，有什么科学道理吗？"
```

**类型3：现象关联型**
```
"吃了一段时间止痛药后血糖好像不太稳，这两者有关系吗？"
```

**类型4：治疗困惑型**
```
"降糖药吃了三个月效果越来越差，是药的问题还是我的问题？"
```

**类型5：生活影响型**
```
"出差一周作息全乱了，回来血糖一直降不下来，需要担心吗？"
```

### 问题长度控制
- **理想长度**：30-80字
- **最大长度**：不超过120字
- **核心要求**：一读就懂患者在问什么

### 避免重复的策略
1. **变换切入角度**：同一推理链可从不同症状/事件切入
2. **变换问法**：直接问因果 vs 请求分析 vs 表达困惑
3. **变换重点**：强调症状 vs 强调诱因 vs 强调时间关系

---

## 答案设计指南

### 答案结构
```
1. 直接回应问题（1-2句）
2. 展开机制解释（核心部分，专业详细）
3. 串联推理链节点（按因果顺序）
4. 给出结论/建议（如适用）
```

### 答案示例
```
"你观察到的现象确实有医学依据。连续熬夜会激活交感神经系统，促使肾上腺释放
皮质醇和肾上腺素，这些激素都有升糖作用——皮质醇会增加肝脏糖异生，肾上腺素
会促进糖原分解。同时，睡眠不足还会降低组织对胰岛素的敏感性。所以即使饮食
没变，几天熬夜后空腹血糖升高3-4个点是完全可以解释的。建议尽快恢复正常作息。"
```

---

## ⚠️ 输出格式（纯JSON，不要有任何markdown标记）

{{
    "query": {{
        "query_type": "multi_hop_clinical_deduction",
        "question": "简洁自然的患者提问（30-80字）",
        "question_style": "问题风格类型（症状疑惑/观察求因/现象关联/治疗困惑/生活影响）",
        "answers": [
            {{
                "content": "专业详细的医学回答（包含机制解释，200-400字）",
                "is_correct": true,
                "explanation": "为什么这是正确答案（评分标准说明）"
            }}
        ],
        "reasoning_chain": [
            {{
                "node_id": 1,
                "session_id": 0,
                "content": "节点内容",
                "role": "起始节点/中间节点/终点节点"
            }}
        ],
        "required_memory_nodes": [
            "Session X: 需要从记忆中召回的关键信息"
        ],
        "source_key_points": [
            {{
                "category": "类别",
                "name": "名称",
                "content": "内容",
                "session_id": 0
            }}
        ],
        "metadata": {{
            "hop_count": 4,
            "reasoning_pattern": "推理模式",
            "question_style": "问题风格",
            "difficulty": "hard",
            "time_span_days": 30,
            "sessions_involved": [1, 5, 10],
            "core_mechanism": "核心机制概述"
        }}
    }},
    "diversity_check": {{
        "different_from_existing": "与已有问题的主要区别",
        "unique_angle": "本问题的独特切入角度"
    }}
}}
"""

# ========== Helper formatting templates ==========

EVENTS_TIMELINE_FORMAT = """
事件ID: {event_id}
日期: {event_date}
类型: {event_type}
内容: {event_content}
触发关系: {triggered_by}
---
"""

KNOWLEDGE_POINTS_BY_SESSION_FORMAT = """
### Session {session_id} ({session_date})
{kps_list}
"""

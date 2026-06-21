"""Query generation prompt templates - refactored version.

New logic:
- EEM: generate fill-in-the-blank from a single KP
- TLA: generate temporal questions from a single KP
- SUA: generate temporal update questions from all KPs in a category
- MQ/IG: keep original logic

Key constraints:
- STRICT JSON format compliance required
- All prompts require pure JSON output without markdown code blocks
"""

# ========== EEM (Entity Exact Match) - single KP fill-in-the-blank ==========

EEM_SINGLE_KP_PROMPT = """You are a professional medical query generation expert. Your task is to generate one **Entity Exact Match (EEM)** fill-in-the-blank question based on a given single knowledge point.

## Current Session ID
{current_session_id}

## Target Knowledge Point
{kp_text}

## Task Description
Based on the knowledge point above, generate **1** entity exact match fill-in-the-blank question.

### EEM Fill-in-the-Blank Design Principles
1. **Extract an entity**: Identify one precise entity/value from the knowledge point content (e.g., drug name, test value, disease name, dosage, etc.)
2. **Construct the question**: Replace the entity with a question form that asks the model to fill in the blank
3. **Unique answer**: The answer must be a precise, unique factual value

### ⚠️ Do NOT Generate Time-Related Questions (Very Important)

**EEM questions must never ask about time! Time-related questions belong to the TLA type!**

**Prohibited question types:**
- ✗ "When did the patient have the test?"
- ✗ "On which date did this event occur?"
- ✗ "What date did the patient start taking the medication?"
- ✗ Any question whose answer is a date/time

**Permitted question types:**
- ✓ "What is the patient's fasting blood glucose level?" (numeric value)
- ✓ "What is the name of the hypoglycemic medication the patient is currently taking?" (drug name)
- ✓ "What class of drug is the patient allergic to?" (allergen)
- ✓ "What is the patient's medication dosage?" (dosage)
- ✓ "What disease has the patient been diagnosed with?" (disease name)
- ✓ "What test does the patient need to undergo?" (test item)

### ⚠️ Answer Format Standards

To ensure answers can be precisely matched, answers must follow these formats:

**Numeric answer formats:**
- Blood glucose: use "X.X mmol/L" format (e.g., "8.2 mmol/L")
- Blood pressure: use "XXX/XX mmHg" format (e.g., "150/95 mmHg")
- Drug dosage: use "XXXmg" or "XXXml" format (e.g., "500mg", "10ml")
- Body weight: use "XXkg" format (e.g., "65kg")

**Medical terminology answer formats:**
- Drug names: use **generic names** (e.g., "metformin" not a brand name)
- Disease names: use **standard medical terminology** (e.g., "type 2 diabetes mellitus" not just "diabetes")
- Test items: use **standard test names** (e.g., "glycated hemoglobin" not "HbA1c")

**Prohibited answer formats:**
- ✗ Dates or times (belongs to TLA type)
- ✗ Colloquial expressions (e.g., "blood sugar is a bit high")
- ✗ Vague expressions (e.g., "around 8-something")
- ✗ Answers with explanations (e.g., "8.2 mmol/L, which is elevated")

### Generation Rules
- Questions should be in natural interrogative form (do not simply blank out words)
- Answers must come from the knowledge point content; do not fabricate
- Answers must be concise and precisely matchable
- Questions should not reveal too much information; maintain an appropriate level of difficulty
- **Absolutely do not ask time-related questions**

### Examples

Knowledge point: Fasting blood glucose | Content: Fasting blood glucose measured at 8.2 mmol/L
Generated question: "What was the patient's most recent fasting blood glucose level?"
Answer: "8.2 mmol/L"

Knowledge point: Medication | Content: Started metformin 500mg twice daily
Generated question: "What is the name of the hypoglycemic medication the patient is currently taking?"
Answer: "metformin"

Knowledge point: Allergy history | Content: Allergic to penicillin-class antibiotics
Generated question: "What class of antibiotic is the patient allergic to?"
Answer: "penicillin-class"

## ⚠️ Output Format Requirements
**Output pure JSON directly. Do not include any additional text, explanation, or markdown formatting.**

{{
    "query": {{
        "query_type": "entity_exact_match",
        "question": "Question content (must not involve time)",
        "answers": [
            {{
                "content": "Precise entity/numeric answer (cannot be a date)",
                "is_correct": true,
                "explanation": "Source explanation for the answer"
            }}
        ],
        "metadata": {{
            "entity_type": "Entity type (medication/disease/test_value/dosage/allergy etc., cannot be date)",
            "entity_value": "Original entity value extracted",
            "answer_format": "Answer format type (numeric/medication_name/disease_name/dosage etc., cannot be date)",
            "difficulty": "Difficulty level (easy/medium/hard)"
        }}
    }}
}}
"""

# ========== TLA (Temporal Localization Accuracy) - single KP temporal ==========

TLA_SINGLE_KP_PROMPT = """You are a professional medical query generation expert. Your task is to generate one **Temporal Localization Accuracy (TLA)** question based on a given single knowledge point.

## Current Session ID
{current_session_id}

## Target Knowledge Point
{kp_text}

## Task Description
Based on the knowledge point above, generate **1** temporal localization question.

### TLA Question Design Principles
1. **Temporal information**: The question must involve time information in the knowledge point
2. **Two question directions**:
   - Ask "At what time did a certain event occur?" (answer is a time)
   - Ask "What event occurred at a certain time?" (answer is the event content)
3. **Precision**: The answer must be verifiable

### Generation Rules
- Time answers should be inferred from both the knowledge point's `time` field and the knowledge point content
- For example, if the content says "yesterday's blood glucose test" and the `time` field is "2024-01-16", the time answer is "2024-01-15"
- Answers must come from the knowledge point content

### Examples

Knowledge point: Blood glucose test | Time: 2024-01-15 | Content: Fasting blood glucose measured today at 8.2 mmol/L
Generated question: "What was the patient's blood glucose test result on 2024-01-15?"
Answer: "Fasting blood glucose 8.2 mmol/L"

Knowledge point: Medication adjustment | Time: 2024-02-03 | Content: Metformin dose adjusted from 500mg to 850mg the day before yesterday
Generated question: "When was the patient's metformin dose adjusted?"
Answer: "2024-02-01"

## ⚠️ Output Format Requirements
**Output pure JSON directly. Do not include any additional text, explanation, or markdown formatting.**

{{
    "query": {{
        "query_type": "temporal_localization",
        "question": "Question content",
        "answers": [
            {{
                "content": "Time or event answer",
                "is_correct": true,
                "explanation": "Source explanation for the answer"
            }}
        ],
        "metadata": {{
            "time_type": "Time type (absolute_time/event_at_time)",
            "target_time": "The time point involved",
            "difficulty": "Difficulty level (easy/medium/hard)"
        }}
    }}
}}
"""

# ========== SUA (State Update Accuracy) - category KP temporal updates ==========

SUA_CATEGORY_PROMPT = """You are a professional medical query generation expert. Your task is to generate one **State Update Accuracy (SUA)** question based on multiple knowledge points from a given category.

## Current Session ID
{current_session_id}

## Category Name
{category}

## All Knowledge Points in This Category (sorted by time)
Total of {kps_count} records:

{kps_text}

## ⚠️ Pre-check (Critically Important)

Before generating a question, you must analyze these knowledge points to determine whether there is a **genuine state change or value update**.

### What is a "genuine state change"?

**✓ Qualifying changes (SUA question can be generated):**
- The same indicator has different values at different times (e.g., blood glucose goes from 11.2 to 8.5 mmol/L)
- The dosage of the same drug is adjusted (e.g., metformin from 500mg to 850mg)
- The severity of the same symptom changes (e.g., hypoglycemic episodes decrease from 2/week to 1/month)
- A change in treatment plan (e.g., switching from oral medication to insulin injection)
- Before/after comparison of test results (e.g., HbA1c drops from 9.2% to 7.5%)

**✗ Non-qualifying (SUA question cannot be generated):**
- Multiple knowledge points describe **different things** with no before/after comparison
- They are just **different aspects** of the same category without any update or conflict
- The information is **newly added**, not **updated** (e.g., first-time recording of an indicator)
- Multiple knowledge points are **essentially the same** content expressed differently

### Decision Flow

1. Read all knowledge points carefully
2. Look for two or more knowledge points that reflect a change or update in a value/state
3. If a change is found, generate a question based on that change point
4. If no change is found, **construct a hypothetical "background fact"** in the question to create a contrast with the existing KPs

## Task Description
If a genuine state change exists, generate **1** state update question.

### ⚠️ SUA Question Design Principles

**Core requirement: Focus on specific change points — do not ask about overall trends!**

SUA questions test whether the model can accurately track the **latest value** or **specific change** of a particular value/state.

**Example question types:**
- ✓ "What is the patient's current (most recent) fasting blood glucose level?" (tracking latest state)
- ✓ "What was the patient's metformin dose adjusted from and to?" (specific change)
- ✓ "What was the patient's most recent blood pressure reading?" (latest value)
- ✓ "The patient took some Weifuchun capsules this month — has their medication regimen changed?" (KPs only show the patient bought omeprazole last month, so a hypothetical Weifuchun scenario is constructed)
- ✓ "The patient's blood glucose was 10.9 mmol/L two weeks ago — what is their most recent reading?" (KPs only show last week's glucose of 12.3 mmol/L, so a hypothetical value is constructed)

### SUA Question Generation Strategies

1. **Prioritize changes in the most recent session**: Identify new states/values appearing in the latest session
2. **Ask "what is the current state"**: Have the model distinguish between past and current states
3. **Ask "how much did it specifically change"**: e.g., "The dose was adjusted from 500mg to what?"

### Answer Format Requirements

Answers must be **precise and matchable**:
- Numeric answers: use standard format (e.g., "8.2 mmol/L", "850mg")
- State answers: clear and unambiguous (e.g., "discontinued", "dose doubled")

### Examples

**Example 1 (correct — asking for latest state):**
Knowledge point list:
[1] Blood glucose | 2024-01-15 | Fasting blood glucose 11.2 mmol/L
[2] Blood glucose | 2024-02-01 | Fasting blood glucose 8.5 mmol/L
[3] Blood glucose | 2024-03-01 | Fasting blood glucose 7.8 mmol/L

Question: "What is the patient's current (most recent) fasting blood glucose level?"
Answer: "7.8 mmol/L"

**Example 2 (correct — asking for specific change):**
Knowledge point list:
[1] Medication | 2024-01-10 | Started metformin 500mg bid
[2] Medication | 2024-02-20 | Metformin dose adjusted to 850mg bid

Question: "The patient's metformin dose was adjusted from 500mg to what?"
Answer: "850mg"

**Example 3 (correct — constructed scenario question):**
Knowledge point list:
[1] Blood glucose | 2024-03-01 | Fasting blood glucose 7.8 mmol/L

Question: "The patient's fasting blood glucose was 8.9 mmol/L on February 12th — what is their most recent fasting blood glucose reading?"
Answer: "7.8 mmol/L"

## ⚠️ Output Format Requirements
**Output pure JSON directly. Do not include any additional text, explanation, or markdown formatting.**

{{
    "query": {{
        "query_type": "state_update",
        "question": "Question content (focused on specific change or latest state)",
        "answers": [
            {{
                "content": "Precise state/numeric answer (concise and matchable)",
                "is_correct": true,
                "explanation": "Source explanation for the answer"
            }}
        ],
        "metadata": {{
            "state_type": "State type (symptom/medication/test_result/treatment_plan/lifestyle)",
            "change_type": "Change type (latest_value/specific_change/recent_update)",
            "focus_session": "Focused session ID (usually the most recent)",
            "difficulty": "Difficulty level (easy/medium/hard)",
            "change_description": "Brief description of the detected state change (e.g., blood glucose dropped from 11.2 to 7.8)"
        }}
    }}
}}
"""

# ========== Common instruction template ==========

KEY_POINTS_STRUCTURE_DESC = """
## Knowledge Point (Key Point) Structure Description

Each knowledge point contains the following fields:
- **category**: Category (test results / physiological indicators / medication records / disease status / user preferences)
- **name**: Key item name (1-4 characters, e.g., "blood glucose", "insulin")
- **content**: Specific content excerpt
- **trap_score**: Difficulty score (0.0-1.0; higher scores indicate greater suitability for question construction)
- **time**: Time of the event
- **session_id**: Source session ID

Knowledge points are cumulative. The same `name` may have multiple records at different times, representing information from different time points.
"""

# ========== Two-stage generation: Phase 1 - Trap Reasoning ==========

TRAP_REASONING_PROMPT = """You are a senior medical exam question design expert skilled at crafting trap questions that can only be answered correctly by integrating the patient's personal history.

## Core Objective
Design a trap scenario where the question **can only be answered correctly by recalling this patient's past information**. Anyone relying on general medical knowledge alone will get it wrong!

## Task Description
Based on a **target knowledge point** and the **full context of its source event**, deeply explore potential medical conflicts and traps between it and the patient's **background information**.

## Target Knowledge Point (Core Basis for This Question)
Category: {target_category}
Name: {target_name}
Content: {target_content}
Time: {target_time}
Source Session: {target_session_id}

## Source Event
The following is the **original event description** corresponding to the session from which the target knowledge point comes:

{source_event_content}

## Patient Background Information (All Historical Knowledge Points)
The following is all known information about this patient. Search for information that may relate to or conflict with the target knowledge point:

{background_kps}

## Trap Reasoning Task

Please think deeply about the following:

### 1. Target Knowledge Point and Source Event Analysis
- What is the core content described by this knowledge point?
- What additional medical details in the **source event** can be leveraged?
- What medical concepts does it involve (drugs, diseases, tests, lifestyle, etc.)?
- What unique question angles can be extracted **from the source event**? (This is key to increasing question diversity!)

### 2. Potential Conflicts and Trap Identification

**Core approach: Identify "hidden conditions" in the background information related to the target knowledge point**

Carefully review the background information and consider the following:

**Time/State change conflicts:**
- Has an important state change occurred recently? (value changes, symptom changes, medication adjustments)
- Is there time-sensitive critical information? (e.g., just took a medication, just completed a test)
- Are there **time points** or **state changes** related to the target knowledge point?

**Drug-related conflicts:**
- Is the patient allergic to any drugs/substances? (e.g., penicillin allergy, iodine allergy, latex allergy)
- Do medications the patient is currently taking interact with any standard treatments?
- Does the patient have medication preferences or contraindications? (e.g., fear of injections, swallowing difficulties)

**Disease-related conflicts:**
- Does the patient's medical history affect certain treatment plans? (e.g., gastric ulcer contraindicates NSAIDs)
- Does the patient's current disease state contradict standard management?

**Lifestyle conflicts:**
- Do the patient's dietary preferences/restrictions conflict with medical advice? (e.g., vegetarian, avoids certain foods)
- Does the patient's financial situation or insurance coverage affect medication choices?
- Do the patient's sleep schedule, work, or exercise habits require special consideration?

### 3. Trap Design Points

Based on the above analysis, design 3-4 high-quality trap descriptions. **Each trap must satisfy:**
1. **Medical knowledge misdirection**: Appears to be correct medical advice but is wrong for this specific patient
2. **Memory dependency**: Requires remembering the patient's specific information to avoid the trap
3. **Professional credibility**: The trap is medically sound, not a low-level error
4. **Diversity**: Prioritize unique trap angles **from the source event**

### Available Trap Types (aim for variety)
- allergy: Allergy-related (drug allergy, food allergy, cross-reactivity)
- drug_interaction: Drug interactions (multi-drug combination risks)
- contraindication: Contraindication-related (disease contraindications, state contraindications)
- preference: Medication/treatment preferences (explicitly stated preferences or aversions)
- lifestyle: Lifestyle conflicts (diet, sleep, financial constraints)
- temporal_change: Time/state changes (recent changes, dosage adjustments)
- dosage_adjustment: Dosage-related (timing of adjustments, cumulative risks)
- symptom_differential: Symptom differentiation (similar symptom confusion, drug side effect misidentification)
- compliance: Adherence-related (actual execution vs. prescribed instructions)
- monitoring: Monitoring-related (blood glucose monitoring, indicator tracking)
- timing: Medication/meal timing (time-sensitive operations)
- economic: Economic factors (cost, insurance coverage, supply costs)

## ⚠️ Output Format Requirements
**Output pure JSON directly. Do not include any additional text, explanation, or markdown formatting.**

{{
    "target_kp_analysis": {{
        "content_summary": "Summary of the core content of the target knowledge point",
        "source_event_insights": "Additional medical points extracted from the source event (to increase diversity)",
        "medical_concepts": ["List of medical concepts involved"],
        "potential_question_angles": ["Possible question angles (prioritize unique angles from the source event)"]
    }},
    "conflict_analysis": {{
        "medication_conflicts": [
            {{
                "conflict": "Specific trap description",
                "related_background": "Relevant content in the background information",
                "trap_potential": "high/medium/low"
            }}
        ],
        "disease_conflicts": [
            {{
                "conflict": "Specific trap description",
                "related_background": "Relevant content in the background information",
                "trap_potential": "high/medium/low"
            }}
        ],
        "lifestyle_conflicts": [
            {{
                "conflict": "Specific trap description",
                "related_background": "Relevant content in the background information",
                "trap_potential": "high/medium/low"
            }}
        ],
        "temporal_conflicts": [
            {{
                "conflict": "Specific trap description",
                "related_background": "Relevant content in the background information",
                "trap_potential": "high/medium/low"
            }}
        ]
    }},
    "trap_points": [
        {{
            "trap_scenario": "Trap scenario: describe a seemingly reasonable but actually unsuitable medical recommendation/choice for this patient",
            "why_trap_works": "Why the trap works: what answer would someone give using only general medical knowledge, and why that answer is wrong",
            "correct_approach": "Correct approach: what the right answer is once the patient's special circumstances are considered",
            "required_memory": ["List of patient information that must be remembered"],
            "trap_type": "Trap type (choose from the types listed above, aim for variety)",
            "difficulty": "hard"
        }}
    ],
    "best_trap_for_question": {{
        "selected_trap_index": 0,
        "reason": "Why this trap was selected as the basis for the question",
        "recommended_question_type": "mq/ig"
    }}
}}
"""


# ========== Two-stage generation: Phase 2 - MQ from trap reasoning ==========

MQ_FROM_TRAP_PROMPT = """You are a senior medical exam question design expert specializing in "memory trap" multiple-choice questions that can only be answered correctly by integrating patient-specific information.

## Core Principles
This question must satisfy ALL of the following requirements:
1. **Memory-dependent**: Without recalling the patient's special information, the answer will be wrong
2. **Professionally deceptive**: All options appear to be reasonable medical recommendations
3. **Concealed trap**: Incorrect options are completely correct standard answers in general situations

## Target Knowledge Point and Corresponding Event Content
Category: {target_category}
Name: {target_name}
Content: {target_content}
Event content: {source_event_content}

## Trap Analysis Results (Must be used!)
{trap_reasoning}

## Patient Background Information Summary
{background_summary}

## Previously Generated Questions
{existing_queries_hint}

---

## ⚠️ Mandatory Trap Type Requirement

You must select one of the following trap types to design the question (prioritize types different from already-generated questions):

### High-Value Trap Types (Use First)

1. **Allergy Cross-Reactivity (allergy_cross)**
   - Patient is allergic to a drug/food → structurally similar substances may also trigger allergy
   - Example: sulfonamide allergy → cross-allergy risk with sulfonylurea hypoglycemics
   - Example: penicillin allergy → caution with cephalosporins
   - Example: latex allergy → certain fruits (banana, avocado) cross-reactivity

2. **Drug Interaction (drug_interaction)**
   - Drugs the patient is currently taking interact with some options
   - Example: taking warfarin → avoid large amounts of vitamin K-rich foods
   - Example: taking metformin → iodine contrast agents require stopping the drug first
   - Example: taking statins → grapefruit juice affects metabolism

3. **Disease State Contraindication (contraindication)**
   - The patient's disease history makes certain options contraindicated
   - Example: gastric ulcer → NSAIDs contraindicated
   - Example: renal impairment → certain drugs require dose adjustment or are prohibited
   - Example: heart failure → certain hypoglycemics (TZDs) contraindicated

4. **Time/State Change (temporal_state)**
   - Recent changes in the patient make standard recommendations no longer applicable
   - Example: just had surgery → some medications temporarily inappropriate
   - Example: dose just adjusted → observation needed, not further adjustment
   - Example: frequent recent hypoglycemia → glycemic targets should be relaxed

5. **Personal Preference/Adherence (preference_compliance)**
   - Preferences or limitations explicitly expressed by the patient
   - Example: needle phobia → injectable regimens not suitable
   - Example: financial difficulty → expensive drugs not suitable
   - Example: irregular work schedule → regimens requiring strict timing not suitable

6. **Value Memory Trap (value_memory)**
   - Requires remembering the patient's specific values (historical vs. most recent)
   - Example: patient's HbA1c dropped from 9% to 7.5% → when asking "current" level, answer should be the latest value
   - Example: glycemic target changed from strict to relaxed → options involve target values

---

## Multiple-Choice Question Design Requirements

### Question Design
- **Format**: Casual oral-style inquiry from the patient (user) to the doctor
- **Length**: Concise and natural, 10-20 words
- **Core**: Must never reveal key information such as disease history, allergy history, or contraindications

**Good question examples:**
- "Doctor, I have a headache — what painkiller should I take?" (no mention of gastric ulcer)
- "Doctor, my blood sugar control isn't great — should I switch medications?" (no mention of allergy history)
- "Doctor, what should I eat for breakfast?" (no mention of food allergy/preference)
- "Doctor, my blood sugar has been fluctuating lately — should I adjust my medication?" (no mention of recent changes)

**Poor question examples:**
- ✗ "I have a gastric ulcer and a headache — what medication should I take?" (reveals key information)
- ✗ "I'm allergic to sulfonamides — can I take glibenclamide?" (reveals the trap)

### Option Design (4 options, 1-3 correct)

**Correct options:**
- Genuinely appropriate recommendations that account for the patient's special circumstances
- Expressed concisely and professionally

**Incorrect options (traps):**
- **In a general situation, they would be completely correct standard medical recommendations**
- They are only inappropriate because of this specific patient's special circumstances
- Expressed equally concisely, professionally, and confidently
- **Never use** language that hints at being wrong (e.g., "use with caution", "possibly", "may carry risks")

### Number of Correct Answers
**Must set 1-3 correct answers based on the actual situation**:
- Do not always default to 2
- Set flexibly based on the needs of the trap design
- Can have only 1 correct answer (all others are traps)
- Can also have 3 correct answers (only 1 trap)

---

## Professional Examples

### Example 1: Allergy Cross-Reactivity Trap (1 correct answer)
**Background**: Patient is allergic to sulfonamide antibiotics
**Question**: Doctor, my blood sugar isn't well controlled — any oral medication recommendations?
A. Glimepiride
B. Metformin
C. Glibenclamide
D. Gliclazide
**Correct answer**: B
**Trap mechanism**: A/C/D are all sulfonylureas, which carry cross-allergy risk with sulfonamides

### Example 2: Disease Contraindication Trap (1 correct answer)
**Background**: Patient has a history of chronic gastric ulcer
**Question**: Doctor, my knee hurts — what anti-inflammatory painkiller should I take?
A. Ibuprofen
B. Acetaminophen
C. Diclofenac sodium
D. Naproxen
**Correct answer**: B
**Trap mechanism**: A/C/D are all NSAIDs, contraindicated in gastric ulcer patients

### Example 3: Drug Interaction Trap (2 correct answers)
**Background**: Patient is currently taking warfarin anticoagulant
**Question**: Doctor, I want to take some vitamins — what's good?
A. Vitamin C tablet
B. Vitamin E soft capsule
C. Vitamin K tablet
D. Vitamin B complex
**Correct answers**: A, D
**Trap mechanism**: C antagonizes warfarin; B may enhance anticoagulation

### Example 4: Temporal State Trap (2 correct answers)
**Background**: Patient just experienced a severe hypoglycemic episode
**Question**: Doctor, my blood sugar is still running high — should I add medication?
A. Hold off and observe; reassess after stabilization
B. Increase insulin dose
C. Relax blood glucose control targets moderately
D. Add another hypoglycemic agent
**Correct answers**: A, C
**Trap mechanism**: Aggressive glucose lowering should be avoided shortly after a hypoglycemic episode

---

## ⚠️ Output Format Requirements
**Output pure JSON directly. Do not include any additional text, explanation, or markdown formatting.**

{{
    "query": {{
        "query_type": "multiple_choice",
        "question": "Concise patient question (must never reveal key medical history)\\n\\nA. Option A\\nB. Option B\\nC. Option C\\nD. Option D",
        "answers": [
            {{
                "content": "A. Option content",
                "is_correct": false,
                "explanation": "Although reasonable in general, not applicable because [specific reason related to this patient]"
            }},
            {{
                "content": "B. Option content",
                "is_correct": true,
                "explanation": "Suitable for this patient because [specific reason]"
            }},
            {{
                "content": "C. Option content",
                "is_correct": false,
                "explanation": "Although reasonable in general, not applicable because [specific reason related to this patient]"
            }},
            {{
                "content": "D. Option content",
                "is_correct": true,
                "explanation": "Suitable for this patient because [specific reason]"
            }}
        ],
        "trap_design": {{
            "trap_type": "Trap type (choose from the 6 types above)",
            "trap_mechanism": "Detailed explanation of how the trap works",
            "why_others_fail": "Why someone without knowledge of the patient's information would choose incorrectly",
            "required_patient_info": ["Required patient information 1", "Required patient information 2"]
        }},
        "source_key_points": [
            {{
                "category": "Category",
                "name": "Knowledge point name",
                "content": "Knowledge point content",
                "session_id": source_session_id
            }}
        ],
        "metadata": {{
            "difficulty": "hard",
            "correct_count": "Number of correct answers (1-3)",
            "trap_count": "Number of trap options"
        }}
    }}
}}
"""


# ========== Two-stage generation: Phase 2 - IG from trap reasoning ==========

IG_FROM_TRAP_PROMPT = """You are a senior medical exam question design expert. Based on the analyzed trap points, generate one **high-quality inference question-and-answer item**.

## Core Principles
This question must be a **trap question that can only be answered correctly by integrating patient-specific information**.
- The question appears to be a routine medical consultation
- If answered using only general medical knowledge, the answer will be wrong or harmful!
- Only by recalling and considering the patient's special circumstances can a correct answer be given
- The question must be designed around the target knowledge point
- **The question should incorporate the patient's experiential context**

## Target Knowledge Point and Corresponding Event Content
Category: {target_category}
Name: {target_name}
Content: {target_content}
Event content: {source_event_content}

## Trap Analysis Results
{trap_reasoning}

## Patient Background Information Summary
{background_summary}

## Previously Generated Questions
{existing_queries_hint}

## Inference Question Design Requirements

### Question Format
- The question is in the form of a **patient (user) consulting a doctor**
- The answer requires reasoning that integrates the patient's special circumstances

### ⚠️ Core Principles for Question Design (Critically Important)

**The question should incorporate events the patient has experienced, simulating the patient's natural oral-style inquiry:**

**Correct examples (contextually grounded, concise and natural, yet sufficiently clinical):**
- "Doctor, I just had a high-carb takeout meal a while back and my blood sugar shot up to over 10. That's happening again lately — what should I do?"
- "Doctor, my recent tests show my HbA1c has dropped quite a bit — do I need to adjust my medication now?"

**Incorrect examples (reveals too much information, strictly prohibited):**
- ✗ "Doctor, I'm a diabetic patient currently taking metformin. I've recently been on corticosteroids for a skin allergy and my blood sugar has gone up — do I need to adjust my hypoglycemic medication?"
- ✗ "I have delayed gastric emptying and I'm on insulin..."

**Key points:**
1. Patients do not proactively disclose their medical history or medication history when asking questions — that is the source of the trap!
2. However, the question can naturally mention the patient's current experience

### ⚠️ Principles for Designing the Standard Wrong Answer (Critically Important)

**The wrong answer must "look completely like the correct answer":**

1. **Never use absolutist language:**
   - ✗ "Absolutely cannot", "must immediately", "strictly prohibited"
   - ✓ Use normal, professional medical advisory tone

2. **The wrong answer must be a "reasonable medical recommendation":**
   - In **ordinary circumstances**, the wrong answer must be a completely correct, standard medical recommendation
   - It is only inapplicable because the patient's special circumstances were not considered
   - Someone who doesn't know the patient's information would find this answer professional and correct

3. **The wrong answer should be expressed professionally and confidently:**
   - Use professional terminology and standard medical expression
   - No hesitation or qualification in tone
   - Make it feel like the "textbook standard answer"

4. **Difference between correct and wrong answers:**
   - The only difference is: whether the patient's special circumstances were considered
   - Both answers should be logically coherent from a medical standpoint

### Examples

**Example 1 (Drug interaction trap):**
Patient background: Diabetes; currently taking corticosteroids for a skin allergy
Question: Doctor, my blood sugar has suddenly risen — should I add medication?

Standard answer (wrong): A rise in blood sugar needs attention. It is recommended to increase the dose of hypoglycemic medication or adjust the treatment regimen to bring blood sugar under control.
Correct answer: Your elevated blood sugar may be related to the corticosteroids you are currently using, which is a common side effect of this medication. It is recommended to monitor closely for now; blood sugar often returns to normal after steroid therapy ends. If it remains persistently high, a short-term adjustment to your regimen can then be considered.

**Example 2 (Disease state trap):**
Patient background: Diabetes; gastroscopy revealed delayed gastric emptying
Question: Doctor, when is the best time to take insulin?

Standard answer (wrong): Rapid-acting insulin is recommended 15-30 minutes before meals, which helps better control the postprandial blood glucose peak.
Correct answer: Because you have delayed gastric emptying, food will be digested and absorbed more slowly. Injecting at the standard time could cause hypoglycemia. It is recommended to inject at the start of a meal or after eating; the exact timing can be adjusted based on blood glucose monitoring results.

**Example 3 (Allergy history trap):**
Patient background: Allergic to sulfonamide-class drugs
Question: Doctor, is gliclazide effective for lowering blood sugar?

Standard answer (wrong): Gliclazide is a sulfonylurea hypoglycemic agent. It is indeed quite effective and is suitable for patients with type 2 diabetes mellitus.
Correct answer: Gliclazide belongs to the sulfonylurea class and shares a similar chemical structure with sulfonamides. Given your allergy history, there is a risk of cross-reactivity. It is recommended to choose a different class of hypoglycemic agent, such as a DPP-4 inhibitor or a GLP-1 receptor agonist.

## ⚠️ Output Format Requirements
**Output pure JSON directly. Do not include any additional text, explanation, or markdown formatting.**

{{
    "query": {{
        "query_type": "inference_generation",
        "question": "Extremely concise patient question (approximately 10 words; must not reveal any medical history information)",
        "answers": [
            {{
                "content": "Correct answer (complete recommendation that accounts for the patient's special circumstances)",
                "is_correct": true,
                "explanation": "Reasoning process: how the correct answer is derived by integrating the patient's special circumstances"
            }}
        ],
        "common_wrong_answer": {{
            "content": "Standard answer (professional, confident, standard medical recommendation that ignores the patient's special circumstances)",
            "why_wrong": "Why this answer is incorrect for this particular patient"
        }},
        "trap_design": {{
            "trap_type": "Trap type",
            "trap_mechanism": "Trap mechanism description",
            "required_patient_info": ["List of patient information that must be remembered"]
        }},
        "source_key_points": [
            {{
                "category": "Category",
                "name": "Knowledge point name",
                "content": "Knowledge point content",
                "session_id": source_session_id
            }}
        ],
        "metadata": {{
            "difficulty": "hard",
            "inference_type": "Inference type"
        }}
    }}
}}
"""


# ========== Legacy prompts (deprecated, kept for compatibility) ==========

EEM_QUERY_PROMPT = EEM_SINGLE_KP_PROMPT  # backward compatibility
TLA_QUERY_PROMPT = TLA_SINGLE_KP_PROMPT  # backward compatibility
SUA_QUERY_PROMPT = SUA_CATEGORY_PROMPT   # backward compatibility
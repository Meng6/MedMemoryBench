# Pipeline 独立数据生成系统

完全独立的批量数据生成系统，用于快速生成大量医疗对话训练数据。

## 📋 系统概述

这个 Pipeline 系统提供三种独立的生成模式：

```
模式 1: 生成用户画像
基础画像模板 → LLM 扩展 → 详细用户背景

模式 2: 生成事件图谱
用户画像 → LLM 生成 → 健康事件时间线（DAG结构）

模式 3: 生成对话会话
画像 + 事件 → 双Agent模拟 → 医患对话记录（按sessions保存）
```

## 🎯 核心特性

- ✅ **三种独立生成模式**：可单独运行，也可组合运行
- ✅ **完全复用现有服务**：使用 `PersonaService`, `EventService`, `DialogueService`
- ✅ **灵活的配置系统**：支持参数化配置和预设配置
- ✅ **按 Sessions 保存对话**：例如 10个画像 × 5个会话 = 50个对话会话
- ✅ **异步并发生成**：高效利用 LLM API
- ✅ **智能跳过机制**：避免重复生成
- ✅ **错误容错和重试**：生产环境就绪
- ✅ **JSON 格式导出**：结构化数据，易于使用

## 📂 项目结构

```
pipeline/
├── __init__.py          # 模块初始化
├── __main__.py          # CLI 入口
├── config.py            # 配置管理（PersonaConfig, EventConfig, DialogueConfig）
├── generator.py         # 核心生成器（DataGenerator 类）
├── exporters.py         # 数据导出器（PersonaExporter, EventExporter, DialogueExporter）
├── cli.py               # 命令行接口
└── README.md            # 使用文档（本文件）
```

## 🚀 快速开始

### 1. 环境配置

确保已配置 `.env` 文件：

```bash
cd backend
cat .env | grep -E "API_KEY|MODEL"
```

### 2. 快速测试

```bash
# 使用预设配置快速测试（3个画像）
python -m pipeline.cli run-all --preset quick

# 预计耗时: 5-10 分钟
# 生成数据: 3个画像 + 3个事件图谱 + 6个对话会话
```

### 3. 查看生成的数据

```bash
# 查看画像数据
cat data/test_personas.json | jq '.personas[0]'

# 查看事件数据
cat data/test_events.json | jq '.event_graphs[0]'

# 查看对话数据（按 sessions）
cat data/test_dialogues.json | jq '.sessions[0]'
```

## 📖 使用指南

### 模式 1: 生成用户画像

将基础画像模板扩展为详细的用户背景信息。

**命令行用法：**

```bash
# 生成前 10 个画像
python -m pipeline.cli generate-personas --count 10

# 生成指定 ID 的画像
python -m pipeline.cli generate-personas --persona-ids "1,2,3,4,5"

# 自定义并发数和输出路径
python -m pipeline.cli generate-personas \
  --count 20 \
  --concurrency 10 \
  --output data/my_personas.json

# 不跳过已存在的画像（重新生成）
python -m pipeline.cli generate-personas \
  --count 5 \
  --no-skip-existing
```

**编程用法：**

```python
import asyncio
from pipeline import DataGenerator, GenerationConfig, PersonaConfig

async def main():
    config = GenerationConfig(
        persona=PersonaConfig(
            persona_ids=[1, 2, 3, 4, 5],
            concurrency=5,
            export_path="data/personas.json"
        )
    )

    generator = DataGenerator(config)
    await generator.initialize()
    result = await generator.generate_personas()

    print(f"生成了 {result['generated']} 个画像")

asyncio.run(main())
```

**输出数据格式：**

```json
{
  "metadata": {
    "export_time": "2024-01-15T10:30:00",
    "total_count": 5
  },
  "personas": [
    {
      "persona_id": 1,
      "base_persona_id": 1,
      "base_info": {
        "type_name": "慢性病管理者",
        "gender": "男",
        "health_goals": ["控制血压", "改善睡眠"]
      },
      "enriched_data": {
        "age_range": "45-55岁",
        "occupation_detail": "中学教师",
        "lifestyle": {...},
        "health_details": {...}
      }
    }
  ]
}
```

### 模式 2: 生成事件图谱

为用户画像生成健康事件时间线（DAG 结构）。

**命令行用法：**

```bash
# 为所有画像生成事件图谱
python -m pipeline.cli generate-events

# 为指定画像生成事件
python -m pipeline.cli generate-events --persona-ids "1,2,3"

# 自定义事件参数
python -m pipeline.cli generate-events \
  --persona-ids "1,2,3,4,5" \
  --start-date "2024-01-01" \
  --time-span 180 \
  --min-events 10 \
  --max-events 20 \
  --output data/my_events.json
```

**编程用法：**

```python
from pipeline import DataGenerator, GenerationConfig, EventConfig

config = GenerationConfig(
    event=EventConfig(
        persona_ids=[1, 2, 3],
        start_date="2024-01-01",
        time_span_days=90,
        min_events=8,
        max_events=15,
        export_path="data/events.json"
    )
)

generator = DataGenerator(config)
await generator.initialize()
result = await generator.generate_events()
```

**输出数据格式：**

```json
{
  "metadata": {
    "total_graphs": 3,
    "total_events": 32
  },
  "event_graphs": [
    {
      "graph_id": 1,
      "persona_id": 1,
      "start_date": "2024-01-01",
      "time_span_days": 90,
      "event_count": 12,
      "events": [
        {
          "event_id": 1,
          "event": "血压升高到145/95",
          "type": "health",
          "event_date": "2024-01-10",
          "triggered_by": []
        },
        {
          "event_id": 2,
          "event": "工作压力增大",
          "type": "work",
          "event_date": "2024-01-15",
          "triggered_by": []
        },
        {
          "event_id": 3,
          "event": "失眠症状出现",
          "type": "health",
          "event_date": "2024-01-20",
          "triggered_by": [1, 2]
        }
      ]
    }
  ]
}
```

### 模式 3: 生成对话会话

生成医患对话交互记录，**按 sessions 组织保存**。

**命令行用法：**

```bash
# 为所有画像生成对话（每个画像 5 个会话）
python -m pipeline.cli generate-dialogues --sessions 5

# 为指定画像生成对话
python -m pipeline.cli generate-dialogues \
  --persona-ids "1,2,3" \
  --sessions 10 \
  --turns 12

# 10个画像 × 5个会话 = 50个对话会话
python -m pipeline.cli generate-dialogues \
  --persona-ids "1,2,3,4,5,6,7,8,9,10" \
  --sessions 5 \
  --turns 10 \
  --output data/50_sessions.json
```

**编程用法：**

```python
from pipeline import DataGenerator, GenerationConfig, DialogueConfig

config = GenerationConfig(
    dialogue=DialogueConfig(
        persona_ids=[1, 2, 3, 4, 5],
        sessions_per_persona=5,  # 每个画像 5 个会话
        max_turns=10,
        allow_natural_end=True,
        export_path="data/dialogues.json"
    )
)

generator = DataGenerator(config)
await generator.initialize()
result = await generator.generate_dialogues()

print(f"生成了 {result['generated']} 个会话")
# 输出: 生成了 25 个会话（5个画像 × 5个会话）
```

**输出数据格式（按 sessions）：**

```json
{
  "metadata": {
    "export_time": "2024-01-15T12:00:00",
    "total_sessions": 25,
    "total_turns": 245,
    "personas_count": 5
  },
  "sessions": [
    {
      "session_id": 1,
      "persona_id": 1,
      "event_id": 5,
      "status": "completed",
      "turn_count": 10,
      "messages": [
        {
          "turn": 1,
          "role": "user",
          "content": "医生您好，我最近经常失眠...",
          "agent_type": "user_agent"
        },
        {
          "turn": 1,
          "role": "assistant",
          "content": "您好，能具体说说失眠的情况吗？",
          "agent_type": "doctor_agent"
        }
      ],
      "knowledge_points": [...],
      "persona_info": {...},
      "event_info": {...}
    }
  ]
}
```

### 完整流程：运行所有阶段

**使用预设配置：**

```bash
# 快速测试（3个画像）
python -m pipeline.cli run-all --preset quick

# 生产环境（所有画像）
python -m pipeline.cli run-all --preset production
```

**编程用法：**

```python
from pipeline import DataGenerator, PRODUCTION_CONFIG

generator = DataGenerator(PRODUCTION_CONFIG)
await generator.initialize()

# 依次运行三个阶段
await generator.generate_personas()
await generator.generate_events()
await generator.generate_dialogues()
```

## 🔧 配置详解

### PersonaConfig（画像配置）

```python
PersonaConfig(
    count=10,                    # 生成数量
    persona_ids=[1,2,3],         # 指定ID（优先级高于 count）
    skip_existing=True,          # 跳过已存在的画像
    concurrency=5,               # 并发数
    max_retries=3,               # 最大重试次数
    export_path="data/personas.json"  # 导出路径
)
```

### EventConfig（事件配置）

```python
EventConfig(
    persona_ids=[1,2,3],         # 要生成事件的画像ID
    skip_existing=True,          # 跳过已有事件图谱的画像
    start_date="2024-01-01",     # 起始日期
    time_span_days=90,           # 时间跨度（天）
    min_events=8,                # 最小事件数
    max_events=15,               # 最大事件数
    concurrency=3,               # 并发数
    export_path="data/events.json"
)
```

### DialogueConfig（对话配置）

```python
DialogueConfig(
    persona_ids=[1,2,3],         # 要生成对话的画像ID
    sessions_per_persona=5,      # 每个画像生成的会话数
    max_turns=10,                # 每个会话的最大轮数
    allow_natural_end=True,      # 允许自然结束
    skip_existing=False,         # 是否跳过已有对话
    concurrency=2,               # 并发数
    export_path="data/dialogues.json",
    export_verbose=True          # 导出详细信息
)
```

### 预设配置

**QUICK_TEST_CONFIG**（快速测试）：
- 3个画像 × 2个会话 = 6个对话会话
- 每个会话最多 5 轮
- 预计耗时：5-10 分钟

**PRODUCTION_CONFIG**（生产环境）：
- 40个画像 × 10个会话 = 400个对话会话
- 每个会话最多 12 轮
- 预计耗时：3-4 小时

## 💡 使用场景示例

### 场景 1: 生成 100 轮对话数据

需求：10个画像 × 10个会话 = 100个会话，每个会话约10轮

```bash
# 方法 1: 分步生成
python -m pipeline.cli generate-personas --count 10
python -m pipeline.cli generate-events --persona-ids "1,2,3,4,5,6,7,8,9,10"
python -m pipeline.cli generate-dialogues \
  --persona-ids "1,2,3,4,5,6,7,8,9,10" \
  --sessions 10 \
  --turns 10

# 方法 2: 编程接口
from pipeline import DataGenerator, GenerationConfig, DialogueConfig

config = GenerationConfig(
    dialogue=DialogueConfig(
        persona_ids=list(range(1, 11)),  # [1,2,3,...,10]
        sessions_per_persona=10,
        max_turns=10
    )
)

generator = DataGenerator(config)
await generator.initialize()
result = await generator.generate_dialogues()
# 结果: 100 个会话，约 1000 轮对话
```

### 场景 2: 为特定画像补充对话

已有 5 个画像，每个画像已有 3 个会话，现在要再补充 7 个会话（达到 10 个）

```bash
python -m pipeline.cli generate-dialogues \
  --persona-ids "1,2,3,4,5" \
  --sessions 10 \
  --skip-existing  # 自动计算需要补充的数量
```

### 场景 3: 批量生成数据集

需求：生成 500 个对话会话的训练数据集

```bash
# 方案：50个画像 × 10个会话 = 500个会话

# 步骤 1: 生成画像
python -m pipeline.cli generate-personas --count 50

# 步骤 2: 生成事件
python -m pipeline.cli generate-events

# 步骤 3: 生成对话
python -m pipeline.cli generate-dialogues --sessions 10

# 查看结果
cat data/generated_dialogues.json | jq '.metadata'
# 输出: {"total_sessions": 500, "total_turns": 4850, ...}
```

## 🔍 数据查看和验证

### 查看画像数据

```bash
# 查看画像总数
cat data/generated_personas.json | jq '.metadata.total_count'

# 查看第一个画像的详细信息
cat data/generated_personas.json | jq '.personas[0]'

# 查看所有画像的年龄分布
cat data/generated_personas.json | jq '.personas[].enriched_data.age_range'
```

### 查看事件数据

```bash
# 查看事件图谱总数和总事件数
cat data/generated_events.json | jq '.metadata'

# 查看第一个事件图谱
cat data/generated_events.json | jq '.event_graphs[0]'

# 统计各类型事件数量
cat data/generated_events.json | \
  jq '[.event_graphs[].events[].type] | group_by(.) | map({type: .[0], count: length})'
```

### 查看对话数据

```bash
# 查看对话统计信息
cat data/generated_dialogues.json | jq '.metadata'

# 查看第一个会话
cat data/generated_dialogues.json | jq '.sessions[0]'

# 统计每个会话的轮数
cat data/generated_dialogues.json | jq '.sessions[].turn_count'

# 提取所有对话消息（用于训练）
cat data/generated_dialogues.json | jq '.sessions[].messages'
```

### 数据库查询

```bash
# 查看数据库中的数据
sqlite3 data/med_eve.db

# 查询画像数量
SELECT COUNT(*) FROM expanded_personas;

# 查询事件图谱数量
SELECT COUNT(*) FROM event_graphs;

# 查询对话会话数量
SELECT COUNT(*) FROM dialogues;

# 查询消息总数
SELECT COUNT(*) FROM messages;
```

## 🐛 常见问题

### 1. ModuleNotFoundError

**错误**：`ModuleNotFoundError: No module named 'app'`

**解决**：确保在 backend 目录下运行
```bash
cd backend
python -m pipeline.cli --help
```

### 2. LLM API 错误

**错误**：`LLM API Error: Invalid API key`

**解决**：检查 .env 配置
```bash
cat .env | grep API_KEY
# 确保 OPENAI_API_KEY 已设置
```

### 3. 数据库锁定

**错误**：`Database is locked`

**解决**：降低并发数
```bash
python -m pipeline.cli generate-dialogues \
  --concurrency 1  # 降低并发
```

### 4. 生成速度慢

**问题**：生成速度比预期慢

**解决**：
- 提高并发数（如果 API 允许）
- 使用更快的模型（在 .env 中配置）
- 检查网络连接

### 5. 重复生成数据

**问题**：不想重复生成已有的数据

**解决**：使用 `--skip-existing` 参数（默认启用）
```bash
python -m pipeline.cli generate-personas --skip-existing
```

## 📊 性能参考

基于 DeepSeek-V3 模型的测试数据：

| 任务 | 数量 | 并发数 | 耗时 | LLM调用次数 |
|-----|------|--------|------|------------|
| 生成画像 | 10个 | 5 | 2-3分钟 | ~10次 |
| 生成事件 | 10个 | 3 | 3-5分钟 | ~10次 |
| 生成对话 | 50个会话 | 2 | 15-20分钟 | ~100次 |
| **完整流程** | 10画像+50会话 | - | **20-30分钟** | ~120次 |

大规模生成参考：
- 100个画像 + 500个会话：约 2-3 小时
- 40个画像 + 400个会话：约 3-4 小时（生产环境配置）

## 🔧 高级用法

### 自定义数据导出

```python
from pipeline.exporters import DialogueExporter
from app.database import AsyncSessionLocal

async def export_custom():
    async with AsyncSessionLocal() as db:
        exporter = DialogueExporter(db)

        # 导出特定画像的对话
        result = await exporter.export(
            persona_ids=[1, 2, 3],
            output_path="data/custom_export.json",
            verbose=True  # 包含详细信息
        )

        print(f"导出了 {result['sessions']} 个会话")
```

### 增量生成

```python
# 第一次生成 5 个会话
config = DialogueConfig(sessions_per_persona=5)
await generator.generate_dialogues(config)

# 后续追加 5 个会话（总共 10 个）
config = DialogueConfig(sessions_per_persona=10, skip_existing=True)
await generator.generate_dialogues(config)
# 只会生成新的 5 个会话
```

### 批量处理多个画像集合

```python
# 分批处理大量画像
persona_batches = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
]

for batch in persona_batches:
    config = DialogueConfig(
        persona_ids=batch,
        sessions_per_persona=10
    )
    await generator.generate_dialogues(config)
```

## 🆘 获取帮助

```bash
# 查看命令帮助
python -m pipeline.cli --help

# 查看子命令帮助
python -m pipeline.cli generate-personas --help
python -m pipeline.cli generate-events --help
python -m pipeline.cli generate-dialogues --help
```

## 📝 许可证

与主项目保持一致。

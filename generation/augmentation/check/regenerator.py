"""Query regenerator for difficulty enhancement."""

import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from .checker import QueryCheckResult
from .enhancer import EnhancementSuggestion

logger = logging.getLogger(__name__)


@dataclass
class RegenerationResult:
    """Result of query regeneration."""

    query_id: str
    success: bool
    original_query: Dict[str, Any]
    new_query: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class QueryRegenerator:
    """Regenerates questions to make them more memory-dependent."""

    def __init__(self, model: Optional[str] = None, temperature: float = 0.9, max_tokens: int = 3000):
        self._load_api_config()
        self.model = model or self._default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Optional[OpenAI] = None

        logger.info(f"[QueryRegenerator] Initialized, model: {self.model}")

    def _load_api_config(self) -> None:
        """Load API configuration from environment."""
        import os
        from dotenv import load_dotenv

        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
        model = os.getenv("LLM_MODEL") or os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
        self._default_model = model.replace("openai/", "") if model.startswith("openai/") else model

    @property
    def client(self) -> OpenAI:
        """Get OpenAI client (lazy init)."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def regenerate_query(
        self,
        original_query: Dict[str, Any],
        enhancement_suggestion: EnhancementSuggestion,
    ) -> RegenerationResult:
        """Regenerate a single query based on enhancement suggestions."""
        query_id = original_query["query_id"]
        query_type = original_query["query_type"]

        prompt = f"""你是一位医学数据集设计专家。请根据以下信息重新生成一个更好的问题。

## 原始问题信息
- **问题类型**: {query_type}
- **原问题**: {enhancement_suggestion.original_question}
- **原答案**: {enhancement_suggestion.expected_answer}

## 问题的源知识点
{json.dumps(original_query.get("source_key_points", []), ensure_ascii=False, indent=2)}

## 问题分析
{enhancement_suggestion.analysis}

## 强化建议
{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(enhancement_suggestion.suggestions))}

## 要求
请重新设计这个问题，使其：
1. 必须依赖患者的特定记忆信息才能回答
2. 不能仅凭医学常识或一般知识就能答对
3. 保持与原问题相同的类型（{query_type}）
4. 答案必须基于源知识点中的信息

请按以下 JSON 格式输出：
```json
{{
    "question": "重新设计的问题",
    "answers": [
        {{
            "content": "正确答案内容",
            "is_correct": true,
            "explanation": "解释为什么这是正确答案"
        }}
    ],
    "metadata": {{
        "regenerated": true,
        "original_query_id": "{query_id}",
        "enhancement_reason": "强化原因简述"
    }}
}}
```
"""

        try:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                )
            except Exception as e:
                if "max_completion_tokens" in str(e) or "max_tokens" in str(e):
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                else:
                    raise
            result_text = response.choices[0].message.content.strip()

            new_query = self._parse_regenerated_query(result_text, original_query)

            if new_query:
                logger.info(f"  [{query_id}] Regeneration succeeded")
                return RegenerationResult(
                    query_id=query_id,
                    success=True,
                    original_query=original_query,
                    new_query=new_query,
                )
            else:
                return RegenerationResult(
                    query_id=query_id,
                    success=False,
                    original_query=original_query,
                    error_message="Failed to parse regenerated query",
                )

        except Exception as e:
            logger.error(f"  [{query_id}] Regeneration failed: {e}")
            return RegenerationResult(
                query_id=query_id,
                success=False,
                original_query=original_query,
                error_message=str(e),
            )

    def _parse_regenerated_query(
        self,
        result_text: str,
        original_query: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Parse regenerated query JSON from LLM response."""
        import re

        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', result_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = result_text

        try:
            parsed = json.loads(json_str)

            new_query = original_query.copy()
            new_query["question"] = parsed.get("question", original_query["question"])
            new_query["answers"] = parsed.get("answers", original_query["answers"])

            original_metadata = original_query.get("metadata", {})
            new_metadata = parsed.get("metadata", {})
            new_query["metadata"] = {**original_metadata, **new_metadata}

            return new_query

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}")
            return None

    def batch_regenerate(
        self,
        queries_to_regenerate: List[Dict[str, Any]],
        suggestions_map: Dict[str, EnhancementSuggestion],
    ) -> List[RegenerationResult]:
        """Batch regenerate queries."""
        results = []

        for query in queries_to_regenerate:
            query_id = query["query_id"]
            suggestion = suggestions_map.get(query_id)

            if suggestion is None:
                logger.warning(f"  [{query_id}] No enhancement suggestion found, skipping")
                continue

            result = self.regenerate_query(query, suggestion)
            results.append(result)

        return results

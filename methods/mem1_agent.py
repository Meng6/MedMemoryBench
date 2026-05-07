"""
MEM1 Agent - Official MEM1 integration for MedMemoryBench.

This agent uses the MEM1 model's core memory consolidation capability via vLLM deployment.
MEM1 maintains a compact internal state using <think> tags to consolidate memory across turns.
"""

import re
import logging
import requests
from typing import Optional, List, Dict, Any

from .base import BaseAgent, MemoryBuildResult, AgentResponse

logger = logging.getLogger(__name__)


class Mem1Agent(BaseAgent):
    """
    MEM1 Agent that leverages the MEM1 model's memory consolidation capability.

    The model is deployed via vLLM and uses <think> tags to maintain internal state.
    Memory is consolidated incrementally during memorize() calls, and the consolidated
    state is used during query() to answer questions.
    """

    METHOD_TYPE = "agentic_memory"

    DEFAULT_VLLM_URL = "http://localhost:8014"
    DEFAULT_MAX_MODEL_LEN = 8192

    def __init__(
        self,
        model: str = "Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        provider: str = "vllm",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        # MEM1 specific parameters
        vllm_url: Optional[str] = None,
        model_path: Optional[str] = None,
        max_state_tokens: int = 4096,
        max_input_tokens: int = 4096,
        max_context_tokens: int = 8192,
        use_chat_api: bool = True,
        **kwargs,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        # vLLM configuration
        self.vllm_url = vllm_url or base_url or self.DEFAULT_VLLM_URL
        self.model_path = model_path or model
        self.use_chat_api = use_chat_api

        # Token limits
        self.max_state_tokens = max_state_tokens
        self.max_input_tokens = max_input_tokens
        self.max_context_tokens = max_context_tokens

        # Internal state - the core of MEM1's memory consolidation
        self._internal_state: str = ""
        self._memory_history: List[str] = []

        # Verify vLLM connection
        self._verify_vllm_connection()

    def _verify_vllm_connection(self) -> bool:
        """Verify that vLLM server is accessible."""
        try:
            response = requests.get(f"{self.vllm_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"vLLM server connected at {self.vllm_url}")
                return True
        except requests.exceptions.RequestException:
            pass

        # Try models endpoint as fallback
        try:
            response = requests.get(f"{self.vllm_url}/v1/models", timeout=5)
            if response.status_code == 200:
                logger.info(f"vLLM server connected at {self.vllm_url}")
                return True
        except requests.exceptions.RequestException:
            pass

        logger.warning(
            f"vLLM server not accessible at {self.vllm_url}. "
            f"Please start vLLM server with: python -m vllm.entrypoints.openai.api_server "
            f"--model {self.model_path} --port 8014"
        )
        return False

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        if not text or max_tokens <= 0:
            return ""
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._tokenizer.decode(tokens[:max_tokens])

    def _split_text_into_chunks(self, text: str, max_tokens: int) -> List[str]:
        """Split text into chunks that fit within token limit."""
        if not text.strip():
            return []
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_text = self._tokenizer.decode(tokens[start:end])
            chunks.append(chunk_text)
            start = end
        return chunks

    def _extract_internal_state(self, response: str) -> Optional[str]:
        """Extract <think>...</think> content from model response."""
        pattern = r"<think>(.*?)</think>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _extract_answer(self, response: str) -> Optional[str]:
        """Extract <answer>...</answer> content from model response."""
        pattern = r"<answer>(.*?)</answer>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _call_vllm_chat(self, messages: List[Dict[str, str]], stop: Optional[List[str]] = None) -> str:
        """Call vLLM server using chat completions API."""
        payload = {
            "model": self.model_path,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": 0.95,
        }
        if stop:
            payload["stop"] = stop

        try:
            response = requests.post(
                f"{self.vllm_url}/v1/chat/completions",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()

            content = result["choices"][0]["message"]["content"].strip()

            # Append stop token if it was the stop reason
            finish_reason = result["choices"][0].get("finish_reason", "")
            stop_reason = result["choices"][0].get("stop_reason", "")

            if stop:
                for s in stop:
                    if s in [finish_reason, stop_reason]:
                        content += s
                        break

            return content

        except requests.exceptions.RequestException as e:
            logger.error(f"vLLM chat API error: {e}")
            raise RuntimeError(f"vLLM server error: {e}")

    def _call_vllm_completion(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call vLLM server using completions API (for continuation generation)."""
        payload = {
            "model": self.model_path,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": 0.95,
            "top_k": -1,
        }
        if stop:
            payload["stop"] = stop

        try:
            response = requests.post(
                f"{self.vllm_url}/v1/completions",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()

            content = result["choices"][0]["text"].strip()

            # Append stop token if it was the stop reason
            stop_reason = result["choices"][0].get("stop_reason", "")
            if stop and stop_reason in stop:
                content += stop_reason

            return content

        except requests.exceptions.RequestException as e:
            logger.error(f"vLLM completion API error: {e}")
            raise RuntimeError(f"vLLM server error: {e}")

    def _consolidate_memory(self, new_content: str) -> str:
        """
        Use MEM1's memory consolidation capability.

        The model uses <think> tags to maintain and update internal state,
        integrating new information with existing memory.
        """
        # Build the consolidation prompt
        if self._internal_state:
            consolidation_prompt = (
                "You are a memory consolidation assistant. Your task is to maintain a compact internal state "
                "that preserves important information for future questions.\n\n"
                "Current internal state:\n"
                f"<think>{self._internal_state}</think>\n\n"
                "New information to integrate:\n"
                f"{new_content}\n\n"
                "Instructions:\n"
                "1. Read the new information carefully\n"
                "2. Identify key facts, entities, relationships, and events\n"
                "3. Update your internal state by integrating new information with existing knowledge\n"
                "4. Remove redundant or outdated information\n"
                "5. Keep the state compact but comprehensive\n\n"
                "Output your updated internal state within <think></think> tags, then confirm with <answer>Memory consolidated.</answer>"
            )
        else:
            consolidation_prompt = (
                "You are a memory consolidation assistant. Your task is to create a compact internal state "
                "that preserves important information for future questions.\n\n"
                "Information to memorize:\n"
                f"{new_content}\n\n"
                "Instructions:\n"
                "1. Read the information carefully\n"
                "2. Identify key facts, entities, relationships, and events\n"
                "3. Create a structured internal state that captures the essential information\n"
                "4. Use bullet points or structured format for clarity\n"
                "5. Keep it compact but comprehensive\n\n"
                "Output your internal state within <think></think> tags, then confirm with <answer>Memory consolidated.</answer>"
            )

        messages = [{"role": "user", "content": consolidation_prompt}]

        try:
            response = self._call_vllm_chat(messages, stop=["</answer>"])
            response += "</answer>" if not response.endswith("</answer>") else ""

            # Extract the new internal state
            new_state = self._extract_internal_state(response)
            if new_state:
                # Truncate if needed
                self._internal_state = self._truncate_to_tokens(new_state, self.max_state_tokens)
                return self._internal_state
            else:
                # Fallback: append to existing state
                fallback_state = (
                    f"{self._internal_state}\n{new_content}" if self._internal_state else new_content
                )
                self._internal_state = self._truncate_to_tokens(fallback_state, self.max_state_tokens)
                return self._internal_state

        except Exception as e:
            logger.warning(f"Memory consolidation failed: {e}, using fallback")
            fallback_state = (
                f"{self._internal_state}\n{new_content}" if self._internal_state else new_content
            )
            self._internal_state = self._truncate_to_tokens(fallback_state, self.max_state_tokens)
            return self._internal_state

    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        """
        Store text into memory using MEM1's consolidation mechanism.

        The text is split into chunks if necessary, and each chunk is consolidated
        into the internal state using the MEM1 model.
        """
        chunks = self._split_text_into_chunks(text, self.max_input_tokens)

        consolidated_states = []
        for chunk in chunks:
            # Consolidate each chunk into internal state
            state = self._consolidate_memory(chunk)
            consolidated_states.append(state[:500] if state else "")
            self._memory_history.append(chunk)

        self._memory_chunks.append(text)
        self._is_initialized = True

        return MemoryBuildResult(
            success=True,
            method="mem1",
            action="consolidate_state",
            input_content=text,
            stored_content=text[:2000],
            memory_entries=[
                {
                    "event": "STATE_UPDATE",
                    "memory": self._internal_state[:500],
                    "chunk_index": i,
                }
                for i, _ in enumerate(chunks)
            ],
            chunk_count=len(self._memory_history),
            extra={
                "state_tokens": self.count_tokens(self._internal_state),
                "chunks_processed": len(chunks),
                "model_path": self.model_path,
                "vllm_url": self.vllm_url,
            },
        )

    def query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> AgentResponse:
        """
        Answer a question using MEM1's reasoning with consolidated memory.

        The internal state (from <think> tags) is provided as context,
        and the model generates an answer using <answer> tags.
        """
        # Build the query prompt with internal state as context
        if self._internal_state:
            query_prompt = (
                f"{system_message}\n\n" if system_message else ""
            ) + (
                "You have the following information in your memory:\n"
                f"<think>{self._internal_state}</think>\n\n"
                "Based on the information above, answer the following question.\n"
                "First, think about what you know in <think></think> tags.\n"
                "Then provide your answer in <answer></answer> tags.\n\n"
                f"Question: {question}"
            )
        else:
            query_prompt = (
                f"{system_message}\n\n" if system_message else ""
            ) + (
                "Answer the following question.\n"
                "First, think step by step in <think></think> tags.\n"
                "Then provide your answer in <answer></answer> tags.\n\n"
                f"Question: {question}"
            )

        messages = [{"role": "user", "content": query_prompt}]

        try:
            response = self._call_vllm_chat(messages, stop=["</answer>"])
            response += "</answer>" if not response.endswith("</answer>") else ""

            # Extract answer
            answer = self._extract_answer(response)
            if not answer:
                # Fallback: use the whole response
                answer = response

            # Extract any updated thinking (for tracking)
            thinking = self._extract_internal_state(response)

            # 构建 retrieved_memories
            retrieved_memories = [
                {
                    "memory": self._internal_state[:1000],
                    "type": "mem1_internal_state",
                    "score": 1.0,
                }
            ] if self._internal_state else []

            return AgentResponse(
                output=answer,
                query_time=0.0,
                retrieved_count=1 if self._internal_state else 0,
                retrieved_memories=retrieved_memories,  # 修复：正确设置字段
                extra={
                    "method": "mem1",
                    "full_response": response,
                    "thinking": thinking[:500] if thinking else None,
                    "state_tokens": self.count_tokens(self._internal_state),
                },
            )

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return AgentResponse(
                output=f"Error: {e}",
                query_time=0.0,
                retrieved_count=0,
                extra={
                    "method": "mem1",
                    "error": str(e),
                },
            )

    def reset(self) -> None:
        """Reset agent state."""
        super().reset()
        self._internal_state = ""
        self._memory_history = []

    def set_context_id(self, context_id: int) -> None:
        """Set context ID, resetting state if context changes."""
        if self._context_id is not None and self._context_id != context_id:
            self.reset()
        super().set_context_id(context_id)

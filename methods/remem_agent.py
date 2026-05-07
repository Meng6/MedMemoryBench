"""ReMem agent adapter for MedMemoryBench.

This adapter integrates ReMem (ICLR 2026) into the evaluation framework,
ensuring all LLM calls go through llm_client for token tracking.

ReMem: Reasoning with Episodic Memory
- Organizes documents into a hybrid memory graph with entities, facts, and episodic gist traces
- Combines dense retrieval with graph exploration for complex question answering
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import AgentResponse, BaseAgent, MemoryBuildResult
from utils.llm_client import (
    BaseLLMClient,
    LLMResponse,
    create_llm_client,
    format_messages,
    get_usage_tracker,
)

logger = logging.getLogger(__name__)


# ============================================================================
# TrackedLLMWrapper - LLM调用包装器
# ============================================================================

class TrackedLLMWrapper:
    """
    包装 ReMem 的 LLM 调用，确保所有调用通过评测框架的 llm_client 进行。
    实现与 CacheOpenAI 相同的接口（infer, batch_infer）。

    这个类模拟 ReMem 的 BaseLLM 接口，但内部使用我们的 llm_client 进行调用，
    从而确保 token 统计的准确性。
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        llm_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        seed: int = 0,
        **kwargs,
    ):
        self.llm_client = llm_client
        self.llm_name = llm_name
        self.temperature = temperature
        self.seed = seed
        self.kwargs = kwargs

        # 模拟 LLMConfig 结构（ReMem 内部使用）
        self.llm_config = _LLMConfigProxy(
            llm_name=llm_name,
            temperature=temperature,
            seed=seed,
        )

    def infer(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Tuple[str, Dict, bool]:
        """
        模拟 CacheOpenAI.infer() 接口。

        Args:
            messages: OpenAI 格式的消息列表
            **kwargs: 额外参数（temperature, response_format等）

        Returns:
            Tuple[response_content, metadata, cache_hit]
        """
        # 合并参数
        temperature = kwargs.get("temperature", self.temperature)

        # 处理 response_format
        extra_kwargs = {}
        if "response_format" in kwargs:
            extra_kwargs["response_format"] = kwargs["response_format"]

        try:
            # 通过 llm_client 调用（自动记录 token）
            response = self.llm_client.chat(
                messages=messages,
                temperature=temperature,
                **extra_kwargs,
            )

            content = response.content

            # 处理 JSON 格式响应
            if kwargs.get("response_format", {}).get("type") == "json_object":
                if content.startswith("```json\n") and content.endswith("```"):
                    content = content[8:-3].strip()

            metadata = {
                "prompt": messages,
                "response": content,
                "prompt_tokens": response.input_tokens,
                "completion_tokens": response.output_tokens,
                "finish_reason": "stop",
            }

            return content, metadata, False  # cache_hit=False

        except Exception as e:
            logger.error(f"LLM inference error: {e}")
            return "", {"error": str(e)}, False

    def batch_infer(
        self,
        messages_list: List[List[Dict[str, str]]],
        max_workers: int = 10,
        **kwargs,
    ) -> List[Tuple[str, Dict, bool]]:
        """
        模拟 CacheOpenAI.batch_infer() 接口。
        使用线程池并行调用。
        """
        results = [None] * len(messages_list)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.infer, msgs, **kwargs): idx
                for idx, msgs in enumerate(messages_list)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Batch infer error at index {idx}: {e}")
                    results[idx] = ("", {"error": str(e)}, False)

        return results


class _LLMConfigProxy:
    """模拟 ReMem 的 LLMConfig 结构"""

    def __init__(self, llm_name: str, temperature: float, seed: int):
        self.generate_params = {
            "model": llm_name,
            "temperature": temperature,
            "seed": seed,
            "n": 1,
        }


# ============================================================================
# TrackedEmbeddingWrapper - Embedding调用包装器
# ============================================================================

class TrackedEmbeddingWrapper:
    """
    包装 ReMem 的 Embedding 调用。
    支持两种模式：
    1. 本地模型：直接调用 sentence-transformers
    2. API调用：通过 OpenAI 兼容接口

    实现与 ReMem 的 BaseEmbeddingModel 相同的接口。
    """

    def __init__(
        self,
        provider: str = "local",
        model: str = "BAAI/bge-small-zh-v1.5",
        model_path: Optional[str] = None,
        dim: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_size: int = 16,
        max_seq_len: int = 512,
        normalize: bool = True,
        **kwargs,
    ):
        self.provider = provider
        self.model_name = model
        self.model_path = model_path or model
        self.dim = dim
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.normalize = normalize

        # ReMem 需要的属性
        self.embedding_model_name = model
        self.embedding_size = dim  # 用于 EmbeddingStore 的后备维度检测

        self._model = None
        self._openai_client = None

        if provider == "local":
            self._init_local_model()
        elif provider in ["openai", "api"]:
            self._init_api_client(api_key, base_url)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def _init_local_model(self):
        """初始化本地 sentence-transformers 模型"""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading local embedding model: {self.model_path}")
            self._model = SentenceTransformer(
                self.model_path,
                trust_remote_code=True,
            )

            # 获取实际维度
            if self.dim is None:
                self.dim = self._model.get_sentence_embedding_dimension()

            # 更新 embedding_size（ReMem 需要）
            self.embedding_size = self.dim

            logger.info(f"Embedding model loaded, dim={self.dim}")

        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

    def _init_api_client(self, api_key: Optional[str], base_url: Optional[str]):
        """初始化 OpenAI 兼容的 API 客户端"""
        from openai import OpenAI

        self._openai_client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "not-needed"),
            base_url=base_url or os.environ.get("EMBEDDING_BASE_URL"),
            timeout=60,
        )
        logger.info(f"Embedding API client initialized, model={self.model_name}")

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        编码文本为嵌入向量。

        Args:
            texts: 文本列表
            **kwargs: instruction 等额外参数

        Returns:
            np.ndarray: 形状为 (len(texts), dim) 的嵌入矩阵
        """
        if isinstance(texts, str):
            texts = [texts]

        # 处理空文本
        texts = [t if t.strip() else "empty" for t in texts]

        # 处理 instruction（ReMem 用于区分 query 和 document）
        instruction = kwargs.get("instruction", "")
        if instruction:
            texts = [f"{instruction}{t}" for t in texts]

        if self.provider == "local":
            embeddings = self._encode_local(texts)
        else:
            embeddings = self._encode_api(texts)

        # 归一化
        if self.normalize and kwargs.get("norm", True):
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms

        return embeddings

    def _encode_local(self, texts: List[str]) -> np.ndarray:
        """使用本地模型编码"""
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=False,  # 我们自己处理归一化
        )
        return np.array(embeddings)

    def _encode_api(self, texts: List[str]) -> np.ndarray:
        """使用 API 编码"""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                response = self._openai_client.embeddings.create(
                    input=batch,
                    model=self.model_name,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Embedding API error: {e}")
                # 返回零向量作为后备
                dim = self.dim or 1536
                all_embeddings.extend([np.zeros(dim) for _ in batch])

        return np.array(all_embeddings)

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """批量编码（与 encode 相同，为了接口兼容）"""
        return self.encode(texts, **kwargs)


# ============================================================================
# RememAgent - 主适配器
# ============================================================================

class RememAgent(BaseAgent):
    """
    ReMem 方法适配器。

    将 ReMem (Reasoning with Episodic Memory) 集成到 MedMemoryBench 评测框架。
    确保所有 LLM 调用通过 llm_client 进行，以便统计 token 用量。

    累积构建模式：
    - 每 session_batch_size 个 session 触发一次图构建
    - 图构建在 memorize() 阶段完成，确保时间统计正确
    - query() 只负责检索和问答
    """

    METHOD_TYPE = "agentic_memory"

    def __init__(
        self,
        # 基础模型配置
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        # ReMem 特定参数
        extract_method: str = "episodic_gist",  # openie / episodic / episodic_gist / temporal
        # 图配置
        is_directed_graph: bool = False,
        synonymy_edge_sim_threshold: float = 0.8,
        synonymy_edge_topk: int = 10,
        # 检索配置
        retrieval_top_k: int = 20,
        qa_top_k: int = 10,
        linking_top_k: int = 5,
        damping: float = 0.5,
        passage_node_weight: float = 0.05,
        # Agent配置
        use_agent: bool = False,
        agent_fixed_tools: bool = False,
        agent_max_steps: int = 5,
        # 缓存配置
        use_cache: bool = True,
        force_index_from_scratch: bool = False,
        save_openie: bool = True,
        # API 并发配置
        extraction_max_workers: int = 5,  # 信息抽取的最大并发数，降低可减少 API 超时
        # 累积构建配置
        # session_batch_size: 累积多少个 sessions 后触发一次图构建
        # 设为 10 以匹配评测框架的 evaluation_interval
        # 设为 1 则禁用累积，每个 session 都立即构建
        session_batch_size: int = 10,
        # Embedding 配置
        embedding_provider: str = "local",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        embedding_model_path: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_batch_size: int = 16,
        embedding_max_seq_len: int = 512,
        # 分块配置
        chunk_size_tokens: int = 8000,
        chunk_overlap_tokens: int = 200,
        # 文本预处理
        text_preprocessor_class_name: str = "TextPreprocessor",
        window_sizes: Optional[List[int]] = None,
        # 最大 token 限制
        max_input_tokens: int = 8000,
        max_context_tokens: int = 120000,
        # 工作目录
        working_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        # 保存配置
        self.provider = provider
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")

        # ReMem 参数
        self.extract_method = extract_method
        self.is_directed_graph = is_directed_graph
        self.synonymy_edge_sim_threshold = synonymy_edge_sim_threshold
        self.synonymy_edge_topk = synonymy_edge_topk
        self.retrieval_top_k = retrieval_top_k
        self.qa_top_k = qa_top_k
        self.linking_top_k = linking_top_k
        self.damping = damping
        self.passage_node_weight = passage_node_weight
        self.use_agent = use_agent
        self.agent_fixed_tools = agent_fixed_tools
        self.agent_max_steps = agent_max_steps
        self.use_cache = use_cache
        self.force_index_from_scratch = force_index_from_scratch
        self.save_openie = save_openie
        self.extraction_max_workers = extraction_max_workers
        self.session_batch_size = session_batch_size

        # Embedding 配置
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_model_path = embedding_model_path
        self.embedding_dim = embedding_dim
        self.embedding_api_key = embedding_api_key
        self.embedding_base_url = embedding_base_url
        self.embedding_batch_size = embedding_batch_size
        self.embedding_max_seq_len = embedding_max_seq_len

        # 分块配置
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.text_preprocessor_class_name = text_preprocessor_class_name
        self.window_sizes = window_sizes or [1, 3]

        # Token 限制
        self.max_input_tokens = max_input_tokens
        self.max_context_tokens = max_context_tokens

        # 工作目录
        self.working_dir = working_dir

        # 初始化 LLM 客户端
        self._llm_client = create_llm_client(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )

        # ReMem 实例池（按 context_id）
        self._remem_instances: Dict[int, Any] = {}

        # 累积缓冲区（按 context_id 存储待处理的 session 内容）
        # 格式: {context_id: [session_text1, session_text2, ...]}
        self._pending_sessions: Dict[int, List[str]] = {}
        # 记录每个 context 的累积 session 数量（用于日志和返回值）
        self._session_counts: Dict[int, int] = {}
        # 标记是否已执行索引（用于检测新的 evaluation unit）
        self._indexed_flags: Dict[int, bool] = {}

        # 设置 ReMem 模块路径
        self._setup_remem_path()
        self._remem_modules_loaded = False

    def _setup_remem_path(self):
        """添加 ReMem src 到 Python 路径"""
        remem_src = Path(__file__).resolve().parent / "REMem" / "src"
        if not remem_src.exists():
            raise ImportError(f"ReMem source folder not found at {remem_src}")

        remem_src_str = str(remem_src)
        if remem_src_str not in sys.path:
            sys.path.insert(0, remem_src_str)

        logger.info(f"ReMem path added: {remem_src_str}")

    def _load_remem_modules(self):
        """延迟加载 ReMem 模块"""
        if self._remem_modules_loaded:
            return

        # 导入 ReMem 核心模块
        from remem.remem import ReMem
        from remem.utils.config_utils import BaseConfig

        self._ReMem = ReMem
        self._BaseConfig = BaseConfig
        self._remem_modules_loaded = True

        logger.info("ReMem modules loaded successfully")

    def _build_remem_config(self, context_id: int) -> Any:
        """
        构建 ReMem 的 BaseConfig 配置对象。

        Args:
            context_id: 上下文ID，用于区分不同的记忆实例

        Returns:
            BaseConfig 实例
        """
        self._load_remem_modules()

        # 确定工作目录
        if self.working_dir:
            save_dir = os.path.join(self.working_dir, f"context_{context_id}")
        else:
            save_dir = os.path.join(
                tempfile.gettempdir(),
                "remem_medmemorybench",
                f"context_{context_id}",
            )
        os.makedirs(save_dir, exist_ok=True)

        # 构建配置
        config = self._BaseConfig(
            # 基础配置
            dataset="medmemorybench",
            save_dir=save_dir,

            # LLM 配置
            llm_name=self.model,
            llm_base_url=self._base_url,
            llm_infer_mode="online",
            temperature=self.temperature,

            # 抽取 LLM（使用相同配置）
            extract_llm_label=self.model,

            # 信息抽取方法
            extract_method=self.extract_method,

            # 图配置
            is_directed_graph=self.is_directed_graph,
            synonymy_edge_sim_threshold=self.synonymy_edge_sim_threshold,
            synonymy_edge_topk=self.synonymy_edge_topk,
            graph_type="facts_and_sim",

            # 检索配置
            retrieval_top_k=self.retrieval_top_k,
            qa_top_k=self.qa_top_k,
            linking_top_k=self.linking_top_k,
            damping=self.damping,
            passage_node_weight=self.passage_node_weight,

            # Agent 配置
            agent_fixed_tools=self.agent_fixed_tools,
            agent_max_steps=self.agent_max_steps,

            # 缓存配置
            force_index_from_scratch=self.force_index_from_scratch,
            force_openie_from_scratch=self.force_index_from_scratch,
            save_openie=self.save_openie,

            # API 并发配置 - 降低并发度可减少 API 超时
            extraction_max_workers=self.extraction_max_workers,

            # Embedding 配置
            embedding_model_name=self.embedding_model,
            embedding_batch_size=self.embedding_batch_size,
            embedding_max_seq_len=self.embedding_max_seq_len,
            embedding_return_as_normalized=True,

            # 文本预处理和分块配置
            # 重要优化：使用 "none" 模式，完全不在 ReMem 内部切分
            # 我们在 agent 层将每个 session 作为一个完整文档传入
            # 这样可以大幅减少 chunk 数量和 LLM 调用次数
            text_preprocessor_class_name=self.text_preprocessor_class_name,
            preprocess_chunk_func="none",  # 不做内部切分
            preprocess_chunk_max_token_size=self.chunk_size_tokens,
            preprocess_chunk_overlap_token_size=self.chunk_overlap_tokens,

            # 评估配置
            do_eval_qa=False,  # 我们用自己的评测
            do_eval_retrieval=False,

            # QA prompt
            qa_passage_prefix="- ",
        )

        return config

    def _create_tracked_remem(self, context_id: int) -> Any:
        """
        创建带有 token 追踪的 ReMem 实例。

        通过替换 LLM 和 Embedding 组件来实现 token 统计。
        """
        self._load_remem_modules()

        # 构建配置
        config = self._build_remem_config(context_id)

        logger.info(f"Creating ReMem instance for context_id={context_id}")
        logger.info(f"  extract_method: {self.extract_method}")
        logger.info(f"  save_dir: {config.save_dir}")

        # 创建包装的 LLM
        tracked_llm = TrackedLLMWrapper(
            llm_client=self._llm_client,
            llm_name=self.model,
            temperature=self.temperature,
        )

        # 创建 ReMem 实例，传入包装的 LLM
        remem = self._ReMem(
            global_config=config,
            llm=tracked_llm,
            extract_llm=tracked_llm,
            qa_llm=tracked_llm,
        )

        # 替换 Embedding 模型
        tracked_embedding = TrackedEmbeddingWrapper(
            provider=self.embedding_provider,
            model=self.embedding_model,
            model_path=self.embedding_model_path,
            dim=self.embedding_dim,
            api_key=self.embedding_api_key,
            base_url=self.embedding_base_url,
            batch_size=self.embedding_batch_size,
            max_seq_len=self.embedding_max_seq_len,
        )
        remem.set_embedding_model(tracked_embedding)

        logger.info(f"ReMem instance created successfully")

        return remem

    def _get_context_id(self) -> int:
        """获取当前上下文ID"""
        return self._context_id if self._context_id is not None else 0

    def _get_remem_instance(self, context_id: int) -> Any:
        """获取或创建指定 context_id 的 ReMem 实例"""
        if context_id not in self._remem_instances:
            self._remem_instances[context_id] = self._create_tracked_remem(context_id)
        return self._remem_instances[context_id]

    def _format_input_documents(self, text: str) -> List[str]:
        """
        将输入文本格式化为 ReMem 期望的文档列表。

        重要优化：不在 agent 层做切分，直接将完整 session 文本作为单个文档。
        切分工作交给 ReMem 内部的 text_preprocessor.batch_preprocess_doc() 统一处理。

        这样可以显著减少 chunk 数量：
        - 旧方式：每个 session 按段落切分 → 200+ chunks (10 sessions × 20 paras)
        - 新方式：每个 session 作为整体 → ReMem 按 chunk_size 统一切分 → 10-20 chunks

        由于 chunk_size_tokens=10240 远大于单个 session 的典型大小(~2000-5000 tokens)，
        大部分 session 会保持完整，不会被切分。
        """
        # 直接返回完整文本作为单个文档
        # ReMem 内部会根据 preprocess_chunk_max_token_size 进行统一切分
        text = text.strip()
        if text:
            return [text]
        return []

    def memorize(self, text: str, is_last_session: bool = False, **kwargs) -> MemoryBuildResult:
        """
        记忆构建阶段。

        设计说明：
        =========
        评测框架通过 is_last_session 参数告诉 Agent 当前是否是 evaluation unit 的最后一个 session。
        - 非最后 session：只累积，不触发构建
        - 最后一个 session：累积后触发图构建

        这样可以：
        1. 正确处理噪声 session（无论多少噪声都会被累积）
        2. 保证图构建在 memorize() 阶段完成，时间统计正确
        3. 每个 evaluation unit 独立构建图

        新 evaluation unit 检测：
        - 通过 _indexed_flags 检测是否进入了新的 unit
        - 如果当前 context 已被索引过，清理旧状态并重新开始
        """
        context_id = self._get_context_id()

        # 设置阶段为 memorize
        get_usage_tracker().set_phase("memorize")

        start_time = time.time()

        # 初始化该 context 的累积缓冲区
        if context_id not in self._pending_sessions:
            self._pending_sessions[context_id] = []
            self._session_counts[context_id] = 0

        # 关键：检测是否需要清理旧状态（进入新的 evaluation unit）
        # 如果当前 context 已经被索引过，说明进入了新的 unit，需要重置
        if self._indexed_flags.get(context_id, False):
            logger.info(f"[ReMem] Context {context_id} was already indexed - new evaluation unit detected")
            logger.info(f"[ReMem] Clearing old state for fresh indexing...")

            # 清理该 context 的旧状态
            if context_id in self._pending_sessions:
                old_count = len(self._pending_sessions[context_id])
                self._pending_sessions[context_id] = []
                logger.info(f"[ReMem] Cleared {old_count} old pending sessions")

            # 清理旧的 ReMem 实例（包含旧的图和嵌入）
            if context_id in self._remem_instances:
                del self._remem_instances[context_id]
                logger.info(f"[ReMem] Cleared old ReMem instance")

            # 重置计数和索引标记
            self._session_counts[context_id] = 0
            self._indexed_flags[context_id] = False

        # 累积 session 内容
        self._pending_sessions[context_id].append(text)
        self._session_counts[context_id] += 1
        session_count = self._session_counts[context_id]
        pending_count = len(self._pending_sessions[context_id])

        # 记录到 memory_chunks（用于统计）
        self._memory_chunks.append(text)

        logger.info(f"[ReMem] Session {session_count} accumulated for context_id={context_id} "
                    f"(text length: {len(text)} chars, pending: {pending_count}, is_last: {is_last_session})")

        # 关键：如果是最后一个 session，触发图构建
        if is_last_session:
            logger.info(f"[ReMem] Last session received, triggering graph construction for {pending_count} sessions...")

            flush_result = self._flush_pending_sessions(context_id)

            if flush_result and flush_result.get("success"):
                time_cost = time.time() - start_time
                graph_info = flush_result.get("graph_info", {})
                openie_count = flush_result.get("openie_count", 0)
                openie_results = flush_result.get("openie_results", [])

                # 从 OpenIE 结果构建 memory_entries
                memory_entries = []
                for doc_result in openie_results:
                    entry = {
                        "content": doc_result.get("passage", doc_result.get("verbatim", ""))[:500],
                    }
                    if "gists" in doc_result:
                        entry["gists"] = doc_result.get("gists", [])[:5]
                    if "facts" in doc_result:
                        entry["facts"] = doc_result.get("facts", [])[:5]
                    if "extracted_entities" in doc_result:
                        entry["entities"] = doc_result.get("extracted_entities", [])[:5]
                    if "extracted_triples" in doc_result:
                        entry["triples"] = doc_result.get("extracted_triples", [])[:5]
                    memory_entries.append(entry)

                return MemoryBuildResult(
                    success=True,
                    method="remem",
                    action="index",
                    input_content=text,
                    stored_content=text,
                    memory_entries=memory_entries,
                    all_passages=[{"content": text[:500]}],
                    chunk_count=len(self._memory_chunks),
                    time_cost=time_cost,
                    extraction_result=json.dumps({
                        "status": "indexed",
                        "session_number": session_count,
                        "batch_sessions_processed": flush_result.get("session_count", 0),
                        "num_documents": flush_result.get("num_documents", 0),
                        "openie_count": openie_count,
                        "graph_info": graph_info,
                        "sample_gists": memory_entries[0].get("gists", [])[:3] if memory_entries else [],
                        "sample_facts": memory_entries[0].get("facts", [])[:3] if memory_entries else [],
                    }, ensure_ascii=False, default=str)[:5000],
                    extra={
                        "context_id": context_id,
                        "extract_method": self.extract_method,
                        "action": "index",
                        "session_number": session_count,
                        "batch_sessions_processed": flush_result.get("session_count", 0),
                        "graph_build_time": flush_result.get("time_cost", 0),
                        "graph_info": graph_info,
                    },
                )
            else:
                # 构建失败
                error_msg = flush_result.get("error", "Unknown error") if flush_result else "Flush returned None"
                time_cost = time.time() - start_time
                return MemoryBuildResult(
                    success=False,
                    method="remem",
                    action="index_failed",
                    input_content=text,
                    stored_content="",
                    memory_entries=[],
                    all_passages=[],
                    chunk_count=len(self._memory_chunks),
                    time_cost=time_cost,
                    extraction_result=f"Error: {error_msg}",
                    extra={
                        "context_id": context_id,
                        "action": "index_failed",
                        "error": error_msg,
                    },
                )
        else:
            # 不是最后一个 session，只返回累积结果
            time_cost = time.time() - start_time

            return MemoryBuildResult(
                success=True,
                method="remem",
                action="accumulate",
                input_content=text,
                stored_content=text,
                memory_entries=[],
                all_passages=[{"content": text[:500]}],
                chunk_count=len(self._memory_chunks),
                time_cost=time_cost,
                extraction_result=json.dumps({
                    "status": "accumulated",
                    "session_number": session_count,
                    "pending_count": pending_count,
                    "message": f"Session {session_count} accumulated. Graph will be built on last session.",
                }, ensure_ascii=False),
                extra={
                    "context_id": context_id,
                    "extract_method": self.extract_method,
                    "action": "accumulate",
                    "session_number": session_count,
                    "pending_sessions": pending_count,
                    "total_accumulated_chars": sum(len(s) for s in self._pending_sessions[context_id]),
                },
            )

    def _flush_pending_sessions(self, context_id: int) -> Optional[Dict[str, Any]]:
        """
        将累积的 sessions 批量处理并构建知识图谱。

        这是实际执行 ReMem index() 的地方。

        Returns:
            构建结果信息，如果没有待处理内容则返回 None
        """
        if context_id not in self._pending_sessions or not self._pending_sessions[context_id]:
            logger.info(f"[ReMem] No pending sessions for context_id={context_id}")
            return None

        pending_texts = self._pending_sessions[context_id]
        session_count = len(pending_texts)

        logger.info(f"[ReMem] Flushing {session_count} accumulated sessions for context_id={context_id}")

        start_time = time.time()

        try:
            # 获取 ReMem 实例
            remem = self._get_remem_instance(context_id)

            # 合并所有累积的 sessions 作为文档列表
            # 每个 session 作为一个独立文档，而非合并成一个
            docs = []
            for session_text in pending_texts:
                formatted_docs = self._format_input_documents(session_text)
                docs.extend(formatted_docs)

            logger.info(f"[ReMem] Combined {session_count} sessions into {len(docs)} documents")

            # 强制从头构建索引
            original_force_index = remem.global_config.force_index_from_scratch
            original_force_openie = remem.global_config.force_openie_from_scratch
            remem.global_config.force_index_from_scratch = True
            remem.global_config.force_openie_from_scratch = True

            # 调用 ReMem 官方 index 方法
            logger.info(f"[ReMem] Calling index() with extract_method={self.extract_method}")
            remem.index(docs)

            # 恢复原始配置
            remem.global_config.force_index_from_scratch = original_force_index
            remem.global_config.force_openie_from_scratch = original_force_openie

            # 关键修复：index 后重置 ready_to_retrieve 标志
            # 确保下次 query 时会重新调用 prepare_retrieval_objects()
            remem.ready_to_retrieve = False
            logger.info(f"[ReMem] Reset ready_to_retrieve=False to refresh retrieval cache on next query")

            # 标记已索引
            self._indexed_flags[context_id] = True

            # 收集记忆构建结果
            graph_info = remem.get_graph_info()

            # 加载 OpenIE 抽取结果
            openie_results = self._load_openie_results(remem)

            time_cost = time.time() - start_time

            # 日志报告
            logger.info(f"[ReMem] Graph construction complete in {time_cost:.2f}s")
            logger.info(f"[ReMem] Graph statistics: {graph_info}")

            if openie_results:
                logger.info(f"[ReMem] OpenIE extraction samples ({len(openie_results)} total):")
                for i, doc in enumerate(openie_results[:2]):
                    logger.info(f"  Sample {i+1}:")
                    logger.info(f"    Passage: {doc.get('passage', doc.get('verbatim', ''))[:100]}...")
                    if "gists" in doc:
                        logger.info(f"    Gists: {doc.get('gists', [])[:3]}")
                    if "facts" in doc:
                        logger.info(f"    Facts: {doc.get('facts', [])[:3]}")

            # 清空累积缓冲区
            self._pending_sessions[context_id] = []
            self._is_initialized = True

            return {
                "success": True,
                "session_count": session_count,
                "num_documents": len(docs),
                "graph_info": graph_info,
                "openie_count": len(openie_results) if openie_results else 0,
                "openie_results": openie_results,
                "time_cost": time_cost,
            }

        except Exception as e:
            logger.error(f"[ReMem] Graph construction error: {e}")
            import traceback
            traceback.print_exc()

            # 清空累积缓冲区，避免重复尝试
            self._pending_sessions[context_id] = []

            return {
                "success": False,
                "session_count": session_count,
                "error": str(e),
                "time_cost": time.time() - start_time,
            }

    def _load_openie_results(self, remem) -> List[Dict]:
        """加载 OpenIE 抽取结果"""
        openie_path = remem.openie_results_path
        if os.path.exists(openie_path):
            try:
                with open(openie_path, 'r', encoding='utf-8') as f:
                    openie_data = json.load(f)
                return openie_data.get("docs", [])
            except Exception as e:
                logger.warning(f"Failed to load OpenIE results: {e}")
        return []

    def query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> AgentResponse:
        """
        查询阶段。

        图应该已经在 memorize() 阶段（最后一个 session 时）构建完成。
        此方法只负责检索和问答。
        """
        context_id = self._get_context_id()

        # 设置阶段为 query
        get_usage_tracker().set_phase("query")

        logger.info(f"[ReMem] Starting query for context_id={context_id}")
        logger.info(f"[ReMem] Question: {question[:100]}...")

        start_time = time.time()

        try:
            # 检查是否已索引
            if not self._indexed_flags.get(context_id, False):
                logger.warning("[ReMem] No data indexed. Please call memorize() first.")
                return AgentResponse(
                    output="Error: No memory data available. Please memorize content first before querying.",
                    query_time=time.time() - start_time,
                    retrieved_count=0,
                    extra={
                        "method": "remem",
                        "error": "no_data_indexed",
                        "message": "No content has been indexed. Call memorize() before query().",
                    },
                )

            # 获取 ReMem 实例
            remem = self._get_remem_instance(context_id)

            if not remem.ready_to_retrieve:
                logger.info("[ReMem] Preparing retrieval objects...")
                remem.prepare_retrieval_objects()

            # 使用 Agent 或 直接检索
            if self.use_agent:
                answer, retrieved_docs, extra_info = self._query_with_agent(
                    remem, question, system_message
                )
            else:
                answer, retrieved_docs, extra_info = self._query_with_rag(
                    remem, question, system_message
                )

            query_time = time.time() - start_time

            logger.info(f"[ReMem] Query complete in {query_time:.2f}s")
            logger.info(f"[ReMem] Retrieved {len(retrieved_docs)} documents")
            logger.info(f"[ReMem] Answer preview: {answer[:100]}...")

            # 转换 retrieved_docs 为 retrieved_memories 格式
            retrieved_memories = [
                {
                    "memory": doc.get("content", "")[:2000],
                    "type": "remem_retrieval",
                    "score": doc.get("score"),
                    "rank": doc.get("rank", i + 1),
                    "source": doc.get("source", "solution_docs"),
                }
                for i, doc in enumerate(retrieved_docs)
            ]
            logger.info(f"[ReMem] Converted to {len(retrieved_memories)} retrieved_memories")

            extra_data = {
                "method": "remem",
                "extract_method": self.extract_method,
                "use_agent": self.use_agent,
                **extra_info,
            }

            return AgentResponse(
                output=answer,
                query_time=query_time,
                retrieved_count=len(retrieved_docs),
                retrieved_memories=retrieved_memories,
                extra=extra_data,
            )

        except Exception as e:
            logger.error(f"[ReMem] Query error: {e}")
            import traceback
            traceback.print_exc()

            return AgentResponse(
                output=f"Error: {str(e)}",
                query_time=time.time() - start_time,
                retrieved_count=0,
                extra={"error": str(e)},
            )

    def _query_with_rag(
        self,
        remem: Any,
        question: str,
        system_message: Optional[str],
    ) -> Tuple[str, List[Dict], Dict]:
        """
        使用标准 RAG 流程进行查询。

        Returns:
            Tuple[answer, retrieved_docs, extra_info]
        """
        # 调用 ReMem 官方的 rag_for_qa 方法
        logger.info(f"[ReMem] Calling rag_for_qa with question: {question[:50]}...")
        solutions, responses, metadata, qa_results, retrieval_results = remem.rag_for_qa(
            queries=[question],
            gold_docs=None,
            gold_answers=None,
            metrics=(),
            to_save=False,
        )

        logger.info(f"[ReMem] rag_for_qa returned: solutions={len(solutions) if solutions else 0}")

        # 提取结果
        if solutions and len(solutions) > 0:
            solution = solutions[0]
            answer = solution.answer if hasattr(solution, 'answer') else str(solution)

            # 调试：打印 solution 的属性
            logger.info(f"[ReMem] Solution type: {type(solution)}")
            logger.info(f"[ReMem] Solution has 'docs': {hasattr(solution, 'docs')}")

            # 提取检索到的文档
            retrieved_docs = []
            if hasattr(solution, 'docs') and solution.docs:
                has_doc_scores = hasattr(solution, 'doc_scores') and solution.doc_scores is not None and len(solution.doc_scores) > 0
                for i, doc in enumerate(solution.docs[:self.qa_top_k]):
                    score = None
                    if has_doc_scores and i < len(solution.doc_scores):
                        score = float(solution.doc_scores[i]) if solution.doc_scores[i] is not None else None
                    retrieved_docs.append({
                        "rank": i + 1,
                        "content": doc[:500] if isinstance(doc, str) else str(doc)[:500],
                        "score": score,
                    })
                logger.info(f"[ReMem] Extracted {len(retrieved_docs)} retrieved_docs from solution.docs")
            else:
                # 备用方案：如果 solution.docs 为空，尝试从 responses 或 embedding store 获取
                logger.warning(f"[ReMem] solution.docs is empty, trying fallback retrieval...")
                retrieved_docs = self._fallback_retrieval(remem, question, responses)

            # 提取图搜索种子（事实三元组）
            graph_seeds = []
            if hasattr(solution, 'graph_seeds') and solution.graph_seeds is not None:
                try:
                    graph_seeds = [str(seed) for seed in solution.graph_seeds[:5]]
                except (TypeError, ValueError):
                    graph_seeds = []

            extra_info = {
                "graph_seeds": graph_seeds,
                "qa_rationale": solution.qa_rationale if hasattr(solution, 'qa_rationale') else None,
            }

            return answer, retrieved_docs, extra_info
        else:
            return "No answer generated", [], {}

    def _query_with_agent(
        self,
        remem: Any,
        question: str,
        system_message: Optional[str],
    ) -> Tuple[str, List[Dict], Dict]:
        """
        使用 GraphAgent 进行多步推理查询。

        注意：这个方法通常不会被调用，因为当使用 episodic_gist 或 temporal
        extract_method 时，ReMem 的 rag_for_qa 方法会自动使用内置的 Agent 策略。
        这个方法仅作为手动启用 Agent 模式的备选方案。

        Returns:
            Tuple[answer, retrieved_docs, extra_info]
        """
        from remem.agent.graph_agent import GraphAgent

        # 准备节点内容字典
        node_chunks_dict = self._prepare_node_chunks_dict(remem)

        # 创建 Agent（配置从 remem.global_config 自动读取）
        agent = GraphAgent(
            llm_model=remem.qa_llm,
            node_chunks_dict=node_chunks_dict,
            remem_instance=remem,
        )

        # 使用正确的 API 进行 Agent 检索
        chunk_ids, chunk_scores, agent_logs = agent.retrieve_with_agent(
            query=question,
            beam_size=self.qa_top_k,
        )

        # 获取检索到的文档内容
        retrieved_docs = []
        if chunk_ids and remem._chunk_embedding_store:
            hash_ids = [remem.passage_node_keys[idx] for idx in chunk_ids[:self.qa_top_k] if idx < len(remem.passage_node_keys)]
            chunk_rows = remem.chunk_embedding_store.get_rows(hash_ids)
            for i, hash_id in enumerate(hash_ids):
                if hash_id in chunk_rows:
                    retrieved_docs.append({
                        "rank": i + 1,
                        "content": chunk_rows[hash_id].get("content", "")[:500],
                        "score": chunk_scores[i] if i < len(chunk_scores) else None,
                    })

        # 提取 Agent 生成的答案
        answer = agent_logs.get("agent_answer", "") if isinstance(agent_logs, dict) else ""

        # 如果 Agent 没有生成答案，使用标准 QA 流程
        if not answer and retrieved_docs:
            # 使用 ReMem 的 QA 生成方法
            from remem.utils.misc_utils import QuerySolution

            docs_content = [d["content"] for d in retrieved_docs]
            solution = QuerySolution(
                question=question,
                docs=docs_content,
                doc_scores=[d.get("score", 0) for d in retrieved_docs],
            )

            # 调用 QA 生成
            solutions, responses, metadata = remem.answer_each_question([solution])
            if solutions and len(solutions) > 0:
                answer = solutions[0].answer if hasattr(solutions[0], 'answer') else str(solutions[0])

        extra_info = {
            "agent_logs": agent_logs if isinstance(agent_logs, dict) else {},
            "chunk_ids": chunk_ids[:10] if chunk_ids else [],
        }

        return answer, retrieved_docs, extra_info

    def _fallback_retrieval(
        self,
        remem: Any,
        question: str,
        responses: Optional[List] = None,
    ) -> List[Dict]:
        """
        备用检索方案：当 solution.docs 为空时，尝试从 responses 或 embedding store 中获取文档。

        这种情况通常发生在：
        1. GraphAgent 返回的节点类型与 return_chunk 参数不匹配
        2. hash_to_index 映射失败
        3. agent 检索过程中出错

        Returns:
            List[Dict]: 检索到的文档列表
        """
        retrieved_docs = []

        # 尝试从 responses 中提取
        if responses and len(responses) > 0:
            response = responses[0]
            if isinstance(response, dict):
                # 检查 agent_chunks
                if "agent_chunks" in response and response["agent_chunks"]:
                    logger.info(f"[ReMem Fallback] Found {len(response['agent_chunks'])} agent_chunks in response")
                    for i, chunk in enumerate(response["agent_chunks"][:self.qa_top_k]):
                        retrieved_docs.append({
                            "rank": i + 1,
                            "content": str(chunk)[:500],
                            "score": None,
                            "source": "response_agent_chunks",
                        })
                    return retrieved_docs

        # 尝试直接从 embedding store 检索（语义搜索）
        try:
            if hasattr(remem, 'episodic_embedding_stores') and remem.episodic_embedding_stores:
                gists_store = remem.episodic_embedding_stores.get("gists")
                if gists_store and hasattr(gists_store, 'search'):
                    logger.info("[ReMem Fallback] Attempting semantic search on gists store...")
                    # 获取 query embedding
                    if hasattr(remem, 'embedding_model') and remem.embedding_model:
                        query_embedding = remem.embedding_model.encode([question])[0]
                        # 执行语义搜索
                        results = gists_store.search(query_embedding, top_k=self.qa_top_k)
                        if results:
                            logger.info(f"[ReMem Fallback] Found {len(results)} results from gists semantic search")
                            for i, result in enumerate(results):
                                content = result.get("content", "") if isinstance(result, dict) else str(result)
                                score = result.get("score", None) if isinstance(result, dict) else None
                                retrieved_docs.append({
                                    "rank": i + 1,
                                    "content": content[:500],
                                    "score": score,
                                    "source": "gists_semantic_search",
                                })
                            return retrieved_docs

                # 如果语义搜索失败，尝试获取所有 gists（作为最后手段）
                if gists_store:
                    all_ids = gists_store.get_all_ids() if hasattr(gists_store, 'get_all_ids') else []
                    if all_ids:
                        logger.info(f"[ReMem Fallback] Getting first {self.qa_top_k} gists from store (total: {len(all_ids)})")
                        rows = gists_store.get_rows(all_ids[:self.qa_top_k])
                        for i, (hash_id, row) in enumerate(rows.items()):
                            content = row.get("content", "") if isinstance(row, dict) else str(row)
                            retrieved_docs.append({
                                "rank": i + 1,
                                "content": content[:500],
                                "score": None,
                                "source": "gists_store_fallback",
                            })
                        return retrieved_docs

        except Exception as e:
            logger.warning(f"[ReMem Fallback] Error during fallback retrieval: {e}")

        logger.warning("[ReMem Fallback] No documents retrieved from any source")
        return retrieved_docs

    def _prepare_node_chunks_dict(self, remem: Any) -> Dict[str, str]:
        """准备 Agent 需要的节点内容字典"""
        node_chunks_dict = {}

        # 添加文档块
        if remem._chunk_embedding_store:
            chunk_rows = remem.chunk_embedding_store.get_text_for_all_rows()
            for hash_id, row in chunk_rows.items():
                node_chunks_dict[hash_id] = row.get("content", "")

        # 添加实体
        if remem._phrase_embedding_store:
            phrase_rows = remem.phrase_embedding_store.get_text_for_all_rows()
            for hash_id, row in phrase_rows.items():
                node_chunks_dict[hash_id] = row.get("content", "")

        return node_chunks_dict

    def reset(self) -> None:
        """重置 Agent 状态"""
        super().reset()
        self._remem_instances = {}
        self._pending_sessions = {}  # 清理累积 session 池
        self._session_counts = {}  # 清理计数
        self._indexed_flags = {}  # 清理索引标记
        logger.info("[ReMem] Agent reset, all instances and pending sessions cleared")

    def set_context_id(self, context_id: int) -> None:
        """设置上下文 ID"""
        super().set_context_id(context_id)
        logger.info(f"[ReMem] Context ID set to {context_id}")

    def get_info(self) -> Dict[str, Any]:
        """获取 Agent 信息"""
        info = super().get_info()
        context_id = self._get_context_id()
        pending_count = len(self._pending_sessions.get(context_id, []))
        is_indexed = self._indexed_flags.get(context_id, False)

        info.update({
            "extract_method": self.extract_method,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "retrieval_top_k": self.retrieval_top_k,
            "qa_top_k": self.qa_top_k,
            "use_agent": self.use_agent,
            "session_batch_size": self.session_batch_size,
            "num_instances": len(self._remem_instances),
            "pending_sessions": {k: len(v) for k, v in self._pending_sessions.items()},
            "is_indexed": is_indexed,
        })
        return info

"""HippoRAG agent adapter for MedMemoryBench.

This adapter integrates HippoRAG 2 (OSU NLP Group) into the evaluation framework,
ensuring all LLM calls go through llm_client for token tracking.

HippoRAG 2: A Powerful Graph-Based RAG Framework
- Uses OpenIE for knowledge graph construction (NER + Triple Extraction)
- Combines dense retrieval with Personalized PageRank for multi-hop reasoning
- Recognition Memory (DSPy Filter) for fact reranking
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

    def __init__(
        self,
        llm_client: BaseLLMClient,
        llm_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        seed: int = 0,
        **kwargs,
    ):
        self.llm_client = llm_client
        self.llm_name = llm_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.kwargs = kwargs

        # 模拟 LLMConfig 结构（HippoRAG 内部使用）
        self.llm_config = _LLMConfigProxy(
            llm_name=llm_name,
            temperature=temperature,
            max_tokens=max_tokens,
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
            **kwargs: 额外参数（temperature, max_completion_tokens等）

        Returns:
            Tuple[response_content, metadata, cache_hit]
        """
        # 合并参数
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_completion_tokens", kwargs.get("max_tokens", self.max_tokens))

        # 处理 response_format
        extra_kwargs = {}
        if "response_format" in kwargs:
            extra_kwargs["response_format"] = kwargs["response_format"]

        try:
            # 通过 llm_client 调用（自动记录 token）
            response = self.llm_client.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra_kwargs,
            )

            content = response.content

            # 处理 JSON 格式响应
            if kwargs.get("response_format", {}).get("type") == "json_object":
                if content.startswith("```json\n") and content.endswith("```"):
                    content = content[8:-3].strip()

            metadata = {
                "prompt_tokens": response.input_tokens,
                "completion_tokens": response.output_tokens,
                "finish_reason": "stop",
            }

            return content, metadata, False  # cache_hit=False

        except Exception as e:
            logger.error(f"LLM inference error: {e}")
            return "", {"error": str(e), "prompt_tokens": 0, "completion_tokens": 0, "finish_reason": "error"}, False


class _LLMConfigProxy:
    """模拟 HippoRAG 的 LLMConfig 结构"""

    def __init__(self, llm_name: str, temperature: float, max_tokens: int, seed: int):
        self.generate_params = {
            "model": llm_name,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "seed": seed,
            "n": 1,
        }


# ============================================================================
# TrackedEmbeddingWrapper - Embedding调用包装器
# ============================================================================

class TrackedEmbeddingWrapper:
    """
    包装 HippoRAG 的 Embedding 调用。
    支持两种模式：
    1. 本地模型：直接调用 sentence-transformers
    2. API调用：通过 OpenAI 兼容接口

    实现与 HippoRAG 的 BaseEmbeddingModel 相同的接口。
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
        dtype: str = "auto",
        **kwargs,
    ):
        self.provider = provider
        self.model_name = model
        self.embedding_model_name = model  # HippoRAG 需要的属性名
        self.model_path = model_path or model
        self.dim = dim
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.normalize = normalize
        self.dtype = dtype
        self.api_key = api_key
        self.base_url = base_url

        # HippoRAG 需要的属性
        self.embedding_size = dim  # 用于 EmbeddingStore 的后备维度检测

        self._model = None
        self._openai_client = None
        self._initialized = False

    def _lazy_init(self):
        """延迟初始化模型"""
        if self._initialized:
            return

        if self.provider == "local":
            self._init_local_model()
        elif self.provider in ["openai", "api"]:
            self._init_api_client()
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

        self._initialized = True

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

            # 更新 embedding_size（HippoRAG 需要）
            self.embedding_size = self.dim

            logger.info(f"Embedding model loaded, dim={self.dim}")

        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

    def _init_api_client(self):
        """初始化 OpenAI 兼容的 API 客户端"""
        from openai import OpenAI

        self._openai_client = OpenAI(
            api_key=self.api_key or os.environ.get("OPENAI_API_KEY", "not-needed"),
            base_url=self.base_url or os.environ.get("EMBEDDING_BASE_URL"),
            timeout=60,
        )
        logger.info(f"Embedding API client initialized, model={self.model_name}")

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        编码文本为嵌入向量。

        Args:
            texts: 文本或文本列表
            **kwargs: instruction 等额外参数

        Returns:
            np.ndarray: 形状为 (len(texts), dim) 的嵌入矩阵
        """
        self._lazy_init()

        if isinstance(texts, str):
            texts = [texts]

        # 处理空文本
        texts = [t if t and t.strip() else "empty" for t in texts]

        # 处理 instruction（HippoRAG 用于区分 query 和 document）
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

    def batch_encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """批量编码（与 encode 相同，为了接口兼容）"""
        return self.encode(texts, **kwargs)


# ============================================================================
# HippoRAGAgent - 主适配器
# ============================================================================

class HippoRAGAgent(BaseAgent):
    """
    HippoRAG 方法适配器。

    将 HippoRAG 2 (OSU NLP Group) 集成到 MedMemoryBench 评测框架。
    确保所有 LLM 调用通过 llm_client 进行，以便统计 token 用量。
    """

    METHOD_TYPE = "graph_rag"

    def __init__(
        self,
        # 基础模型配置
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        # HippoRAG 特定参数
        openie_mode: str = "online",  # "online" | "offline"
        # 图配置
        is_directed_graph: bool = False,
        synonymy_edge_sim_threshold: float = 0.8,
        synonymy_edge_topk: int = 2047,
        # 检索配置
        linking_top_k: int = 5,
        retrieval_top_k: int = 200,
        qa_top_k: int = 5,
        damping: float = 0.5,
        passage_node_weight: float = 0.05,
        # 缓存配置
        force_index_from_scratch: bool = False,
        force_openie_from_scratch: bool = False,
        save_openie: bool = True,
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

        # HippoRAG 参数
        self.openie_mode = openie_mode
        self.is_directed_graph = is_directed_graph
        self.synonymy_edge_sim_threshold = synonymy_edge_sim_threshold
        self.synonymy_edge_topk = synonymy_edge_topk
        self.linking_top_k = linking_top_k
        self.retrieval_top_k = retrieval_top_k
        self.qa_top_k = qa_top_k
        self.damping = damping
        self.passage_node_weight = passage_node_weight
        self.force_index_from_scratch = force_index_from_scratch
        self.force_openie_from_scratch = force_openie_from_scratch
        self.save_openie = save_openie

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

        # HippoRAG 实例池（按 context_id）
        self._hipporag_instances: Dict[int, Any] = {}

        # 累积缓冲区（按 context_id 存储待处理的 session 内容）
        # 格式: {context_id: [session_text1, session_text2, ...]}
        self._pending_sessions: Dict[int, List[str]] = {}
        # 记录每个 context 的累积 session 数量（用于日志和返回值）
        self._session_counts: Dict[int, int] = {}
        # 标记是否已执行索引（用于检测新的 evaluation unit）
        self._indexed_flags: Dict[int, bool] = {}

        # 设置 HippoRAG 模块路径
        self._setup_hipporag_path()
        self._hipporag_modules_loaded = False

        # 共享的 Embedding 模型实例
        self._shared_embedding_model = None

    def _setup_hipporag_path(self):
        """添加 HippoRAG src 到 Python 路径并mock缺失的模块"""
        hipporag_src = Path(__file__).resolve().parent / "HippoRAG" / "src"
        if not hipporag_src.exists():
            raise ImportError(f"HippoRAG source folder not found at {hipporag_src}")

        hipporag_src_str = str(hipporag_src)
        if hipporag_src_str not in sys.path:
            sys.path.insert(0, hipporag_src_str)

        # Mock vllm module if not installed (only needed for offline mode)
        if 'vllm' not in sys.modules:
            try:
                import vllm
            except ImportError:
                # Create mock vllm module
                import types
                mock_vllm = types.ModuleType('vllm')
                mock_vllm.SamplingParams = type('SamplingParams', (), {})
                mock_vllm.LLM = type('LLM', (), {})
                sys.modules['vllm'] = mock_vllm

                # Mock vllm.model_executor submodules
                mock_model_executor = types.ModuleType('vllm.model_executor')
                sys.modules['vllm.model_executor'] = mock_model_executor

                mock_guided_decoding = types.ModuleType('vllm.model_executor.guided_decoding')
                sys.modules['vllm.model_executor.guided_decoding'] = mock_guided_decoding

                mock_guided_fields = types.ModuleType('vllm.model_executor.guided_decoding.guided_fields')
                mock_guided_fields.GuidedDecodingRequest = type('GuidedDecodingRequest', (), {})
                sys.modules['vllm.model_executor.guided_decoding.guided_fields'] = mock_guided_fields

                logger.info("vllm module mocked (offline mode not available)")

        logger.info(f"HippoRAG path added: {hipporag_src_str}")

    def _load_hipporag_modules(self):
        """延迟加载 HippoRAG 模块"""
        if self._hipporag_modules_loaded:
            return

        # 导入 HippoRAG 核心模块
        from hipporag.HippoRAG import HippoRAG
        from hipporag.utils.config_utils import BaseConfig

        self._HippoRAG = HippoRAG
        self._BaseConfig = BaseConfig
        self._hipporag_modules_loaded = True

        logger.info("HippoRAG modules loaded successfully")

    def _get_shared_embedding_model(self) -> TrackedEmbeddingWrapper:
        """获取共享的 Embedding 模型实例"""
        if self._shared_embedding_model is None:
            self._shared_embedding_model = TrackedEmbeddingWrapper(
                provider=self.embedding_provider,
                model=self.embedding_model,
                model_path=self.embedding_model_path,
                dim=self.embedding_dim,
                api_key=self.embedding_api_key,
                base_url=self.embedding_base_url,
                batch_size=self.embedding_batch_size,
                max_seq_len=self.embedding_max_seq_len,
            )
        return self._shared_embedding_model

    def _build_hipporag_config(self, context_id: int) -> Any:
        """
        构建 HippoRAG 的 BaseConfig 配置对象。

        Args:
            context_id: 上下文ID，用于区分不同的记忆实例

        Returns:
            BaseConfig 实例
        """
        self._load_hipporag_modules()

        # 确定工作目录
        if self.working_dir:
            save_dir = os.path.join(self.working_dir, f"context_{context_id}")
        else:
            save_dir = os.path.join(
                tempfile.gettempdir(),
                "hipporag_medmemorybench",
                f"context_{context_id}",
            )
        os.makedirs(save_dir, exist_ok=True)

        # 为embedding模型名称添加Transformers/前缀，使HippoRAG能识别
        # HippoRAG会根据这个前缀选择TransformersEmbeddingModel
        # 但我们之后会用我们自己的TrackedEmbeddingWrapper替换它
        # 优先使用本地路径（如果提供了的话），避免从HuggingFace下载
        embedding_model_id = self.embedding_model_path if self.embedding_model_path else self.embedding_model
        hipporag_embedding_name = f"Transformers/{embedding_model_id}"

        # 构建配置
        config = self._BaseConfig(
            # 基础配置
            dataset=None,  # 自由模式
            save_dir=save_dir,

            # LLM 配置
            llm_name=self.model,
            llm_base_url=self._base_url,
            temperature=self.temperature,
            max_new_tokens=self.max_tokens,

            # 信息抽取模式
            openie_mode=self.openie_mode,

            # 图配置
            is_directed_graph=self.is_directed_graph,
            synonymy_edge_sim_threshold=self.synonymy_edge_sim_threshold,
            synonymy_edge_topk=self.synonymy_edge_topk,

            # 检索配置
            linking_top_k=self.linking_top_k,
            retrieval_top_k=self.retrieval_top_k,
            qa_top_k=self.qa_top_k,
            damping=self.damping,
            passage_node_weight=self.passage_node_weight,

            # 缓存配置
            force_index_from_scratch=self.force_index_from_scratch,
            force_openie_from_scratch=self.force_openie_from_scratch,
            save_openie=self.save_openie,

            # Embedding 配置 - 使用Transformers/前缀让HippoRAG识别
            embedding_model_name=hipporag_embedding_name,
            embedding_batch_size=self.embedding_batch_size,
            embedding_max_seq_len=self.embedding_max_seq_len,
            embedding_return_as_normalized=True,
        )

        return config

    def _create_tracked_hipporag(self, context_id: int) -> Any:
        """
        创建带有 token 追踪的 HippoRAG 实例。

        通过替换 LLM 和 Embedding 组件来实现 token 统计。
        """
        self._load_hipporag_modules()

        # 构建配置
        config = self._build_hipporag_config(context_id)

        logger.info(f"Creating HippoRAG instance for context_id={context_id}")
        logger.info(f"  openie_mode: {self.openie_mode}")
        logger.info(f"  save_dir: {config.save_dir}")
        logger.info(f"  embedding_model: {self.embedding_model}")

        # 创建包装的 LLM
        tracked_llm = TrackedLLMWrapper(
            llm_client=self._llm_client,
            llm_name=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # 获取共享的 Embedding 模型
        tracked_embedding = self._get_shared_embedding_model()

        # 创建 HippoRAG 实例
        # 我们需要在创建后替换其内部组件
        hipporag = self._HippoRAG(global_config=config)

        # 关键：替换 LLM 模型
        hipporag.llm_model = tracked_llm

        # 替换 OpenIE 中的 LLM
        if hasattr(hipporag, 'openie') and hipporag.openie is not None:
            hipporag.openie.llm_model = tracked_llm

        # 替换 DSPyFilter 中的 LLM 调用函数
        if hasattr(hipporag, 'rerank_filter') and hipporag.rerank_filter is not None:
            hipporag.rerank_filter.llm_infer_fn = tracked_llm.infer

        # 替换 Embedding 模型
        hipporag.embedding_model = tracked_embedding
        if hasattr(hipporag, 'chunk_embedding_store') and hipporag.chunk_embedding_store is not None:
            hipporag.chunk_embedding_store.embedding_model = tracked_embedding
        if hasattr(hipporag, 'entity_embedding_store') and hipporag.entity_embedding_store is not None:
            hipporag.entity_embedding_store.embedding_model = tracked_embedding
        if hasattr(hipporag, 'fact_embedding_store') and hipporag.fact_embedding_store is not None:
            hipporag.fact_embedding_store.embedding_model = tracked_embedding

        logger.info(f"HippoRAG instance created successfully")

        return hipporag

    def _get_context_id(self) -> int:
        """获取当前上下文ID"""
        return self._context_id if self._context_id is not None else 0

    def _get_hipporag_instance(self, context_id: int) -> Any:
        """获取或创建指定 context_id 的 HippoRAG 实例"""
        if context_id not in self._hipporag_instances:
            self._hipporag_instances[context_id] = self._create_tracked_hipporag(context_id)
        return self._hipporag_instances[context_id]

    def _format_input_documents(self, text: str) -> List[str]:
        """
        将输入文本格式化为 HippoRAG 期望的文档列表。

        HippoRAG 的 index() 方法期望接收文档列表。
        HippoRAG 内部会通过 chunk_embedding_store 处理分块，
        这里只需简单按段落分割即可。
        """
        # 按段落分割，保持简单
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if paragraphs:
            return paragraphs
        # 如果没有段落分隔符，返回整个文本
        return [text] if text.strip() else []

    def _load_openie_results(self, hipporag) -> List[Dict]:
        """加载 OpenIE 抽取结果"""
        openie_path = hipporag.openie_results_path
        if os.path.exists(openie_path):
            try:
                with open(openie_path, 'r', encoding='utf-8') as f:
                    openie_data = json.load(f)
                return openie_data.get("docs", [])
            except Exception as e:
                logger.warning(f"Failed to load OpenIE results: {e}")
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
            logger.info(f"[HippoRAG] Context {context_id} was already indexed - new evaluation unit detected")
            logger.info(f"[HippoRAG] Clearing old state for fresh indexing...")

            # 清理该 context 的旧状态
            if context_id in self._pending_sessions:
                old_count = len(self._pending_sessions[context_id])
                self._pending_sessions[context_id] = []
                logger.info(f"[HippoRAG] Cleared {old_count} old pending sessions")

            # 清理旧的 HippoRAG 实例（包含旧的图和嵌入）
            if context_id in self._hipporag_instances:
                del self._hipporag_instances[context_id]
                logger.info(f"[HippoRAG] Cleared old HippoRAG instance")

            # 重置计数和索引标记
            self._session_counts[context_id] = 0
            self._indexed_flags[context_id] = False

        # 累积 session 内容
        self._pending_sessions[context_id].append(text)
        self._session_counts[context_id] += 1
        session_count = self._session_counts[context_id]
        pending_count = len(self._pending_sessions[context_id])

        logger.info(f"[HippoRAG] Session {session_count} accumulated for context_id={context_id} "
                    f"(text length: {len(text)} chars, pending: {pending_count}, is_last: {is_last_session})")

        # 关键：如果是最后一个 session，触发图构建
        if is_last_session:
            logger.info(f"[HippoRAG] Last session received, triggering graph construction for {pending_count} sessions...")

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
                        "content": doc_result.get("passage", "")[:500],
                        "entities": doc_result.get("extracted_entities", []),
                        "triples": doc_result.get("extracted_triples", [])[:5],
                    }
                    memory_entries.append(entry)

                return MemoryBuildResult(
                    success=True,
                    method="hipporag",
                    action="index",
                    input_content=text,
                    stored_content=text,
                    memory_entries=memory_entries,
                    all_passages=[{"content": text[:500]}],
                    chunk_count=session_count,
                    time_cost=time_cost,
                    extraction_result=json.dumps({
                        "status": "indexed",
                        "session_number": session_count,
                        "batch_sessions_processed": flush_result.get("session_count", 0),
                        "num_documents": flush_result.get("num_documents", 0),
                        "openie_count": openie_count,
                        "graph_info": graph_info,
                        "sample_entities": memory_entries[0].get("entities", [])[:5] if memory_entries else [],
                        "sample_triples": memory_entries[0].get("triples", [])[:3] if memory_entries else [],
                    }, ensure_ascii=False, default=str)[:5000],
                    extra={
                        "context_id": context_id,
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
                    method="hipporag",
                    action="index_failed",
                    input_content=text,
                    stored_content="",
                    memory_entries=[],
                    all_passages=[],
                    chunk_count=session_count,
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
                method="hipporag",
                action="accumulate",
                input_content=text,
                stored_content=text,
                memory_entries=[],
                all_passages=[{"content": text[:500]}],
                chunk_count=session_count,
                time_cost=time_cost,
                extraction_result=json.dumps({
                    "status": "accumulated",
                    "session_number": session_count,
                    "pending_count": pending_count,
                    "message": f"Session {session_count} accumulated. Graph will be built on last session.",
                }, ensure_ascii=False),
                extra={
                    "context_id": context_id,
                    "action": "accumulate",
                    "session_number": session_count,
                    "pending_sessions": pending_count,
                    "total_accumulated_chars": sum(len(s) for s in self._pending_sessions[context_id]),
                },
            )

    def _flush_pending_sessions(self, context_id: int) -> Optional[Dict[str, Any]]:
        """
        将累积的 sessions 批量处理并构建知识图谱。

        这是实际执行 HippoRAG index() 的地方。

        Returns:
            构建结果信息，如果没有待处理内容则返回 None
        """
        if context_id not in self._pending_sessions or not self._pending_sessions[context_id]:
            logger.info(f"[HippoRAG] No pending sessions for context_id={context_id}")
            return None

        pending_texts = self._pending_sessions[context_id]
        session_count = len(pending_texts)

        logger.info(f"[HippoRAG] Flushing {session_count} accumulated sessions for context_id={context_id}")

        start_time = time.time()

        try:
            # 获取 HippoRAG 实例
            hipporag = self._get_hipporag_instance(context_id)

            # 合并所有累积的 sessions
            combined_text = "\n\n".join(pending_texts)

            # 格式化输入文档
            docs = self._format_input_documents(combined_text)
            logger.info(f"[HippoRAG] Combined {session_count} sessions: {len(combined_text)} chars, "
                        f"split into {len(docs)} documents")

            # 调用 HippoRAG 官方 index 方法
            logger.info(f"[HippoRAG] Calling index() with openie_mode={self.openie_mode}")
            hipporag.index(docs)

            # 关键修复：index 后重置 ready_to_retrieve 标志
            # 确保下次 query 时会重新调用 prepare_retrieval_objects()
            # 以加载新增的 passages 和 embeddings
            hipporag.ready_to_retrieve = False
            logger.info(f"[HippoRAG] Reset ready_to_retrieve=False to refresh retrieval cache on next query")

            # 收集记忆构建结果
            graph_info = hipporag.get_graph_info()

            # 加载 OpenIE 抽取结果
            openie_results = self._load_openie_results(hipporag)

            time_cost = time.time() - start_time

            # 日志报告
            logger.info(f"[HippoRAG] Graph construction complete in {time_cost:.2f}s")
            logger.info(f"[HippoRAG] Graph statistics:")
            logger.info(f"  - Phrase nodes: {graph_info.get('num_phrase_nodes', 0)}")
            logger.info(f"  - Passage nodes: {graph_info.get('num_passage_nodes', 0)}")
            logger.info(f"  - Extracted triples: {graph_info.get('num_extracted_triples', 0)}")
            logger.info(f"  - Total nodes: {graph_info.get('num_total_nodes', 0)}")
            logger.info(f"  - Total triples: {graph_info.get('num_total_triples', 0)}")

            if openie_results:
                logger.info(f"[HippoRAG] OpenIE extraction samples ({len(openie_results)} total):")
                for i, doc in enumerate(openie_results[:3]):
                    logger.info(f"  Sample {i+1}:")
                    logger.info(f"    Passage: {doc.get('passage', '')[:100]}...")
                    logger.info(f"    Entities: {doc.get('extracted_entities', [])[:5]}")
                    logger.info(f"    Triples: {doc.get('extracted_triples', [])[:2]}")

            # 清空累积缓冲区
            self._pending_sessions[context_id] = []
            # 标记已索引
            self._indexed_flags[context_id] = True
            self._is_initialized = True

            return {
                "success": True,
                "session_count": session_count,
                "num_documents": len(docs),
                "graph_info": graph_info,
                "openie_count": len(openie_results) if openie_results else 0,
                "openie_results": openie_results,  # 添加 OpenIE 结果以供 memorize() 使用
                "time_cost": time_cost,
            }

        except Exception as e:
            logger.error(f"[HippoRAG] Graph construction error: {e}")
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

        logger.info(f"[HippoRAG] Starting query for context_id={context_id}")
        logger.info(f"[HippoRAG] Question: {question[:100]}...")

        start_time = time.time()

        try:
            # 检查是否已索引
            if not self._indexed_flags.get(context_id, False):
                logger.warning("[HippoRAG] No data indexed. Please call memorize() first.")
                return AgentResponse(
                    output="Error: No memory data available. Please memorize content first before querying.",
                    query_time=time.time() - start_time,
                    retrieved_count=0,
                    extra={
                        "method": "hipporag",
                        "error": "no_data_indexed",
                        "message": "No content has been indexed. Call memorize() before query().",
                    },
                )

            # 获取 HippoRAG 实例
            hipporag = self._get_hipporag_instance(context_id)

            # 检查是否有数据可供检索
            passage_count = len(hipporag.chunk_embedding_store.get_all_ids())
            if passage_count == 0:
                logger.warning("[HippoRAG] No data indexed. Please call memorize() first.")
                return AgentResponse(
                    output="Error: No memory data available. Please memorize content first before querying.",
                    query_time=time.time() - start_time,
                    retrieved_count=0,
                    extra={
                        "method": "hipporag",
                        "error": "no_data_indexed",
                        "message": "No content has been indexed. Call memorize() before query().",
                    },
                )

            # 确保检索对象准备就绪
            if not hipporag.ready_to_retrieve:
                logger.info("[HippoRAG] Preparing retrieval objects...")
                hipporag.prepare_retrieval_objects()

            # 调用 HippoRAG 官方的 rag_qa 方法
            solutions, responses, metadata = hipporag.rag_qa(queries=[question])

            # 提取结果
            if solutions and len(solutions) > 0:
                solution = solutions[0]
                answer = solution.answer if hasattr(solution, 'answer') else str(solution)

                # 提取检索到的文档
                retrieved_docs = []
                if hasattr(solution, 'docs') and solution.docs:
                    for i, doc in enumerate(solution.docs[:self.qa_top_k]):
                        doc_info = {
                            "rank": i + 1,
                            "content": doc[:500] if isinstance(doc, str) else str(doc)[:500],
                        }
                        if hasattr(solution, 'doc_scores') and solution.doc_scores is not None and len(solution.doc_scores) > i:
                            doc_info["score"] = float(solution.doc_scores[i])
                        retrieved_docs.append(doc_info)
            else:
                answer = "No answer generated"
                retrieved_docs = []

            query_time = time.time() - start_time

            logger.info(f"[HippoRAG] Query complete in {query_time:.2f}s")
            logger.info(f"[HippoRAG] Retrieved {len(retrieved_docs)} documents")
            logger.info(f"[HippoRAG] Answer preview: {answer[:100]}...")

            # 转换 retrieved_docs 为 retrieved_memories 格式
            retrieved_memories = [
                {
                    "content": doc.get("content", "")[:500] if isinstance(doc, dict) else str(doc)[:500],
                    "rank": doc.get("rank", i + 1) if isinstance(doc, dict) else i + 1,
                    "score": doc.get("score", 0.0) if isinstance(doc, dict) else 0.0,
                }
                for i, doc in enumerate(retrieved_docs)
            ]

            return AgentResponse(
                output=answer,
                query_time=query_time,
                retrieved_count=len(retrieved_docs),
                retrieved_memories=retrieved_memories,
                extra={
                    "method": "hipporag",
                    "openie_mode": self.openie_mode,
                    "retrieved_docs": retrieved_docs[:5],
                    "retrieval_times": {
                        "ppr_time": hipporag.ppr_time if hasattr(hipporag, 'ppr_time') else 0,
                        "rerank_time": hipporag.rerank_time if hasattr(hipporag, 'rerank_time') else 0,
                    },
                },
            )

        except Exception as e:
            logger.error(f"[HippoRAG] Query error: {e}")
            import traceback
            traceback.print_exc()

            return AgentResponse(
                output=f"Error: {str(e)}",
                query_time=time.time() - start_time,
                retrieved_count=0,
                extra={"error": str(e)},
            )

    def reset(self) -> None:
        """重置 Agent 状态"""
        super().reset()
        self._hipporag_instances = {}
        self._pending_sessions = {}
        self._session_counts = {}
        self._indexed_flags = {}  # 清理索引标记
        logger.info("[HippoRAG] Agent reset, all instances and pending sessions cleared")

    def set_context_id(self, context_id: int) -> None:
        """设置上下文 ID"""
        super().set_context_id(context_id)
        logger.info(f"[HippoRAG] Context ID set to {context_id}")

    def get_info(self) -> Dict[str, Any]:
        """获取 Agent 信息"""
        info = super().get_info()
        context_id = self._get_context_id()
        is_indexed = self._indexed_flags.get(context_id, False)

        info.update({
            "openie_mode": self.openie_mode,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "linking_top_k": self.linking_top_k,
            "retrieval_top_k": self.retrieval_top_k,
            "qa_top_k": self.qa_top_k,
            "damping": self.damping,
            "num_instances": len(self._hipporag_instances),
            "pending_sessions": {k: len(v) for k, v in self._pending_sessions.items()},
            "is_indexed": is_indexed,
        })
        return info

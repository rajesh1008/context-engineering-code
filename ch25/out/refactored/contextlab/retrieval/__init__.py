"""contextlab.retrieval: Chapter 12's from-scratch retrieval stack.

Rungs: keyword (BM25), dense (bi-encoder stand-in), hybrid (fusion),
reranked (cross-encoder stand-in), graph-augmented (link expansion).
Everything is deterministic and runs on the CPU in seconds.
"""
from .text import tokenize, analyse
from .index import InvertedIndex
from .dense import SimEncoder, DenseIndex
from .fusion import reciprocal_rank_fusion, weighted
from .rerank import SimCrossEncoder
from .hnsw import HNSW
from .metrics import hit_at_k, recall_at_k, mrr, ndcg_at_k

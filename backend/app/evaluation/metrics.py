"""Evaluation metrics for retrieval and generation."""
import math
from collections import defaultdict


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Recall@K."""
    if not relevant_ids:
        return 0.0
    retrieved_at_k = set(retrieved_ids[:k])
    return len(retrieved_at_k & relevant_ids) / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank."""
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def hit_rate(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Hit Rate (1 if any relevant item is retrieved, 0 otherwise)."""
    return 1.0 if any(rid in relevant_ids for rid in retrieved_ids) else 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """NDCG@K."""
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)

    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def compute_retrieval_metrics(
    all_results: list[dict],
) -> dict:
    """Compute aggregate retrieval metrics."""
    recall_1_scores = []
    recall_3_scores = []
    recall_5_scores = []
    recall_10_scores = []
    mrr_scores = []
    hit_scores = []
    ndcg_scores = []

    for item in all_results:
        retrieved_ids = [r["chunk_id"] for r in item.get("results", [])]
        relevant_ids = set(item.get("relevant_chunk_ids", []))

        if not relevant_ids:
            continue

        recall_1_scores.append(recall_at_k(retrieved_ids, relevant_ids, 1))
        recall_3_scores.append(recall_at_k(retrieved_ids, relevant_ids, 3))
        recall_5_scores.append(recall_at_k(retrieved_ids, relevant_ids, 5))
        recall_10_scores.append(recall_at_k(retrieved_ids, relevant_ids, 10))
        mrr_scores.append(mrr(retrieved_ids, relevant_ids))
        hit_scores.append(hit_rate(retrieved_ids, relevant_ids))
        ndcg_scores.append(ndcg_at_k(retrieved_ids, relevant_ids, 10))

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "recall@1": round(avg(recall_1_scores), 4),
        "recall@3": round(avg(recall_3_scores), 4),
        "recall@5": round(avg(recall_5_scores), 4),
        "recall@10": round(avg(recall_10_scores), 4),
        "mrr": round(avg(mrr_scores), 4),
        "hit_rate": round(avg(hit_scores), 4),
        "ndcg@10": round(avg(ndcg_scores), 4),
        "num_questions": len(all_results),
        "questions_with_relevance": sum(1 for item in all_results if item.get("relevant_chunk_ids")),
    }

from __future__ import annotations

from dataclasses import dataclass
import random

from sklearn.metrics.pairwise import cosine_similarity

from core.config import BridgeProtRetrievalConfig
from retrieval.bank import BridgeProtDemoExample
from data.schema import Dialogue


@dataclass(slots=True)
class _DialogueSignature:
    num_turns: int
    num_speakers: int
    avg_chars: float


class BridgeProtExampleSelector:
    def __init__(
        self,
        examples: list[BridgeProtDemoExample],
        *,
        config: BridgeProtRetrievalConfig,
    ) -> None:
        if not examples:
            raise ValueError("Few-shot example bank is empty.")
        self.examples = examples
        self.config = config
        self.rng = random.Random(config.seed)
        self.example_signatures = [
            _DialogueSignature(
                num_turns=example.num_turns,
                num_speakers=example.num_speakers,
                avg_chars=_average_chars(example.serialized_dialogue),
            )
            for example in examples
        ]
        self._embedder = None
        self._embeddings = None

        if config.strategy in {"semantic", "hybrid"}:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(config.embedding_model)
            self._embeddings = self._embedder.encode(
                [example.serialized_dialogue for example in examples],
                normalize_embeddings=True,
                show_progress_bar=False,
            )

    def select(
        self,
        *,
        dialogue: Dialogue,
        serialized_dialogue: str,
        num_shots: int | None = None,
    ) -> list[BridgeProtDemoExample]:
        ranked_examples = self.rank(
            dialogue=dialogue,
            serialized_dialogue=serialized_dialogue,
        )
        target_k = num_shots or self.config.num_shots
        return ranked_examples[:target_k]

    def rank(
        self,
        *,
        dialogue: Dialogue,
        serialized_dialogue: str,
    ) -> list[BridgeProtDemoExample]:
        candidate_indices = [
            index
            for index, example in enumerate(self.examples)
            if not (
                example.dataset == dialogue.dataset
                and example.split == dialogue.split
                and example.dialogue_id == dialogue.dialogue_id
            )
        ]
        if not candidate_indices:
            return []

        strategy = self.config.strategy.lower()
        if strategy == "random":
            selected = list(candidate_indices)
            self.rng.shuffle(selected)
            return [self.examples[index] for index in selected]

        semantic_scores = self._semantic_scores(serialized_dialogue, candidate_indices)
        if strategy == "semantic":
            ranked = sorted(candidate_indices, key=lambda idx: semantic_scores[idx], reverse=True)
            return [self.examples[index] for index in ranked]

        if strategy == "hybrid":
            query_signature = _DialogueSignature(
                num_turns=dialogue.num_utterances,
                num_speakers=len({utterance.speaker for utterance in dialogue.utterances}),
                avg_chars=_average_chars(serialized_dialogue),
            )
            ranked = sorted(
                candidate_indices,
                key=lambda idx: self._hybrid_score(
                    semantic_scores[idx],
                    query_signature=query_signature,
                    example_signature=self.example_signatures[idx],
                ),
                reverse=True,
            )
            return [self.examples[index] for index in ranked]

        raise ValueError(
            f"Unknown retrieval strategy '{self.config.strategy}'. "
            "Expected one of: random, semantic, hybrid."
        )

    def _semantic_scores(self, serialized_dialogue: str, candidate_indices: list[int]) -> dict[int, float]:
        if self._embedder is None or self._embeddings is None:
            raise RuntimeError("Semantic retrieval requested but no embedding model is initialized.")
        query_embedding = self._embedder.encode(
            [serialized_dialogue],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores = cosine_similarity(query_embedding, self._embeddings)[0]
        return {index: float(scores[index]) for index in candidate_indices}

    def _hybrid_score(
        self,
        semantic_score: float,
        *,
        query_signature: _DialogueSignature,
        example_signature: _DialogueSignature,
    ) -> float:
        turn_score = _bounded_similarity(query_signature.num_turns, example_signature.num_turns)
        speaker_score = _bounded_similarity(query_signature.num_speakers, example_signature.num_speakers)
        char_score = _bounded_similarity(query_signature.avg_chars, example_signature.avg_chars)
        return float(0.7 * semantic_score + 0.15 * turn_score + 0.10 * speaker_score + 0.05 * char_score)


def _average_chars(serialized_dialogue: str) -> float:
    lines = [line for line in serialized_dialogue.splitlines() if line.strip()]
    if not lines:
        return 0.0
    return float(sum(len(line) for line in lines) / len(lines))


def _bounded_similarity(left: float, right: float) -> float:
    denom = max(abs(left), abs(right), 1.0)
    return float(max(0.0, 1.0 - abs(left - right) / denom))

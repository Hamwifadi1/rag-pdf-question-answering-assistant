"""Store PDF chunk embeddings and retrieve similar chunks with FAISS."""

from typing import TypedDict

import faiss
import numpy as np

from src.chunker import TextChunk
from src.embeddings import EmbeddingModel, EmbeddingResult, generate_text_embedding


DEFAULT_TOP_K = 4


class VectorStoreError(Exception):
    """Raised when a FAISS index cannot be built or queried."""


class RetrievedChunk(TypedDict):
    """Chunk metadata returned by similarity search."""

    chunk_id: int
    text: str
    page_number: int
    similarity_score: float


class RetrievalResult(TypedDict):
    """Retrieved chunks and question-embedding debug metadata."""

    chunks: list[RetrievedChunk]
    question_embedding_dimension: int


class FAISSVectorStore:
    """In-memory cosine-similarity index with aligned chunk metadata."""

    def __init__(
        self,
        index: faiss.Index,
        chunks: list[TextChunk],
        embedding_dimension: int,
    ) -> None:
        self.index = index
        self.chunks = chunks
        self.embedding_dimension = embedding_dimension

    @property
    def vector_count(self) -> int:
        """Return the number of vectors stored in FAISS."""
        return self.index.ntotal

    @classmethod
    def from_embeddings(
        cls,
        chunks: list[TextChunk],
        embedding_result: EmbeddingResult,
    ) -> "FAISSVectorStore":
        """Build a FAISS index while preserving chunk position metadata."""
        embeddings = embedding_result["embeddings"]
        dimension = embedding_result["embedding_dimension"]

        if not chunks or not embeddings:
            raise VectorStoreError("Chunks and embeddings are required to build FAISS.")
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Chunk and embedding counts do not match.")
        if dimension <= 0:
            raise VectorStoreError("Embedding dimension must be greater than zero.")

        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim != 2 or vectors.shape != (len(chunks), dimension):
            raise VectorStoreError("Embedding vectors have an invalid shape.")
        if not np.isfinite(vectors).all():
            raise VectorStoreError("Embedding vectors contain invalid numeric values.")

        # Embeddings are normalized, so inner product equals cosine similarity.
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)

        return cls(index=index, chunks=list(chunks), embedding_dimension=dimension)

    def retrieve(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        model: EmbeddingModel | None = None,
    ) -> RetrievalResult:
        """Return up to top_k chunks most similar to the user question."""
        if top_k <= 0:
            raise VectorStoreError("top_k must be greater than zero.")
        if self.vector_count == 0:
            raise VectorStoreError("The FAISS index is empty.")

        question_embedding = generate_text_embedding(question, model=model)
        question_dimension = len(question_embedding)
        if question_dimension != self.embedding_dimension:
            raise VectorStoreError(
                "Question embedding dimension does not match the FAISS index."
            )

        query_vector = np.asarray([question_embedding], dtype="float32")
        faiss.normalize_L2(query_vector)
        result_count = min(top_k, self.vector_count)
        scores, positions = self.index.search(query_vector, result_count)

        retrieved: list[RetrievedChunk] = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue
            chunk = self.chunks[int(position)]
            retrieved.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "page_number": chunk["page_number"],
                    "similarity_score": float(score),
                }
            )

        return {
            "chunks": retrieved,
            "question_embedding_dimension": question_dimension,
        }

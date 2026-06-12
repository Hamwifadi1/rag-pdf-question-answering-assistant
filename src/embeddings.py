"""Generate free local embeddings for PDF text chunks."""

from functools import lru_cache
from typing import Protocol, TypedDict

from src.chunker import TextChunk


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingResult(TypedDict):
    """Embedding vectors and debug metadata for a chunk batch."""

    embeddings: list[list[float]]
    chunk_count: int
    embedding_dimension: int


class EmbeddingError(Exception):
    """Raised when embeddings cannot be generated or validated."""


class EmbeddingModel(Protocol):
    """Minimal sentence-transformers model interface used for testing."""

    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> object:
        """Encode text into embedding vectors."""


@lru_cache(maxsize=1)
def load_embedding_model() -> EmbeddingModel:
    """Load and cache the local sentence-transformers embedding model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise EmbeddingError(
            "sentence-transformers is not installed. Run: "
            "pip install -r requirements.txt"
        ) from error

    try:
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception as error:
        raise EmbeddingError(
            "Could not load the local embedding model. The first run needs "
            "an internet connection to download the model; later runs use "
            "the local cache."
        ) from error


def generate_chunk_embeddings(
    chunks: list[TextChunk],
    model: EmbeddingModel | None = None,
) -> EmbeddingResult:
    """Generate one normalized local embedding per chunk."""
    if not chunks:
        raise EmbeddingError("No text chunks are available for embedding.")

    texts = [chunk["text"].strip() for chunk in chunks]
    if any(not text for text in texts):
        raise EmbeddingError("All chunks must contain non-empty text.")

    if model is None:
        model = load_embedding_model()

    try:
        encoded = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as error:
        raise EmbeddingError(
            "The local embedding model could not generate embeddings."
        ) from error

    try:
        embeddings = encoded.tolist()
    except AttributeError:
        embeddings = [list(vector) for vector in encoded]

    if len(embeddings) != len(chunks):
        raise EmbeddingError(
            "The model returned a different number of embeddings than expected."
        )
    if not embeddings or not embeddings[0]:
        raise EmbeddingError("The model returned empty embedding vectors.")

    embedding_dimension = len(embeddings[0])
    if any(len(vector) != embedding_dimension for vector in embeddings):
        raise EmbeddingError("The model returned inconsistent embedding dimensions.")

    return {
        "embeddings": embeddings,
        "chunk_count": len(chunks),
        "embedding_dimension": embedding_dimension,
    }


def generate_text_embedding(
    text: str,
    model: EmbeddingModel | None = None,
) -> list[float]:
    """Generate one normalized embedding for a question or other text."""
    clean_text = text.strip()
    if not clean_text:
        raise EmbeddingError("The question cannot be empty.")

    if model is None:
        model = load_embedding_model()

    try:
        encoded = model.encode(
            [clean_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as error:
        raise EmbeddingError(
            "The local embedding model could not embed the question."
        ) from error

    try:
        vectors = encoded.tolist()
    except AttributeError:
        vectors = [list(vector) for vector in encoded]

    if len(vectors) != 1 or not vectors[0]:
        raise EmbeddingError("The model returned an invalid question embedding.")

    return vectors[0]

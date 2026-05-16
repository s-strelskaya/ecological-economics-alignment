"""
converting text to dense vectors using Sentence-BERT.
"""

import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

# default model: fast, good semantic similarity performance
DEFAULT_MODEL = "all-MiniLM-L6-v2"

class CorpusEmbedder:
    
    ### wrapping Sentence-BERT to embed a list of texts and a reference
    # document (the Aims & Scope) into a shared vector space

    ### embeddings are L2-normalised -> 
    # cosine similarity reduces to a simple dot product
    
    
    # parameters: 
    """
    model_name : str
        Sentence-BERT model name: defaults to 'all-MiniLM-L6-v2'
    cache_dir : str | Path | None
        directory to save/load precomputed embeddings
        if None -> no caching is performed and embeddings are recomputed every time
    """

    # examples:
    """
    embedder = CorpusEmbedder(cache_dir="data")
    abstract_vecs = embedder.embed_corpus(abstracts)
    scope_vec = embedder.embed_scope(scope_text)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir  = Path(cache_dir) if cache_dir else None
        self._model: SentenceTransformer | None = None  # lazy load

    
    ###### PUBLIC INTERFACE ######

    def embed_corpus(
        self,
        texts: list[str],
        batch_size: int = 64,
        force_recompute: bool = False,
    ) -> np.ndarray:
        
        # embedding a list of texts (abstracts)

        ### if a cache file exists and force_recompute is False -> 
        # the cached vectors are loaded instead of recomputing.

        # parameters
        """
        texts: list[str]
            the abstract strings to embed
        batch_size: int
            number of texts processed per forward pass (default 64)
        force_recompute: bool
            if True: ignore any cached embeddings (default False)
        """

        # should return
        """
        np.ndarray
            shape (n_texts, embedding_dim), L2-normalised
        """
        
        cache_path = self._cache_path("abstract_vecs.npy")

        if cache_path and cache_path.exists() and not force_recompute:
            log.info("Loading cached abstract embeddings from %s", cache_path)
            return np.load(cache_path)

        log.info("Embedding %d abstracts with %s ...", len(texts), self.model_name)
        vecs = self._model_instance().encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=batch_size,
        )
        self._maybe_save(vecs, cache_path)
        return vecs

    def embed_scope(
        self,
        scope_text: str,
        force_recompute: bool = False,
    ) -> np.ndarray:
        
        # embedding the Aims & Scope reference text

        # parameters
        """
        scope_text: str
            the full Aims & Scope string
        force_recompute: bool
            if True -> ignore any cached embedding (default False)
        """

        # should return
        """
        np.ndarray
            shape (embedding_dim,), L2-normalised
        """
        
        cache_path = self._cache_path("scope_vec.npy")

        if cache_path and cache_path.exists() and not force_recompute:
            log.info("Loading cached scope embedding from %s", cache_path)
            return np.load(cache_path)

        log.info("Embedding Aims & Scope text ...")
        vec = self._model_instance().encode(
            scope_text,
            normalize_embeddings=True,
        )
        self._maybe_save(vec, cache_path)
        return vec

    
    ###### PRIVATE HELPERS ######

    def _model_instance(self) -> SentenceTransformer:
        # lazy-load the model on first use
        if self._model is None:
            log.info("Loading Sentence-BERT model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            log.info("Model loaded. Embedding dimension: %d",
                     self._model.get_sentence_embedding_dimension())
        return self._model

    def _cache_path(self, filename: str) -> Path | None:
        if self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / filename

    @staticmethod
    def _maybe_save(arr: np.ndarray, path: Path | None) -> None:
        if path is not None:
            np.save(path, arr)
            log.info("Saved embeddings to %s", path)

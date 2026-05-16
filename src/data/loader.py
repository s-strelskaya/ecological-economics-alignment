"""
loading and validating paper corpus and Aims & Scope text
"""

import json
import logging
from pathlib import Path
import pandas as pd

log = logging.getLogger(__name__)


class CorpusLoader:
    
    ### loading paper corpus from a .jsonl file and the Aims & Scope
    # reference text from a .txt file; validating both on load

    # parameters
    """
    papers_path: str | Path
        path to the .jsonl file produced by fetcher.py
    scope_path: str | Path
        path to the plain-text Aims & Scope file
    min_abstract_len: int
        papers with abstracts shorter than this are dropped (default 80)
    """

    # examples
    """
    loader = CorpusLoader("data/raw/papers.jsonl", "data/aims_scope.txt")
    df = loader.load_papers()
    scope = loader.load_scope()
    """

    def __init__(
        self,
        papers_path: str | Path,
        scope_path: str | Path,
        min_abstract_len: int = 80,
    ) -> None:
        self.papers_path = Path(papers_path)
        self.scope_path = Path(scope_path)
        self.min_abstract_len = min_abstract_len

    
    ##### PUBLIC INTERFACE #####

    def load_papers(self) -> pd.DataFrame:
        """
        ### loading papers from .jsonl, dropping entries without a usable abstract,
        # and returning a clean df
        """

        # should return
        """
        pd.DataFrame
            columns: title, doi, date, year, abstract, abstract_len
        """
        self._check_exists(self.papers_path)
        raw = self._read_jsonl(self.papers_path)

        df = pd.DataFrame(raw)
        df["year"] = df["year"].astype(int)
        df["abstract_len"] = df["abstract"].str.len()

        before = len(df)
        df = df[df["abstract_len"] >= self.min_abstract_len].reset_index(drop=True)
        dropped = before - len(df)

        log.info(
            "Loaded %d papers (%d dropped — abstract too short). Years: %d–%d.",
            len(df), dropped, df["year"].min(), df["year"].max(),
        )
        return df

    def load_scope(self) -> str:
        
        # loading and returning Aims & Scope text

        # should return
        """
        str
            cleaned scope text (stripped of leading/trailing whitespace)
        """
        self._check_exists(self.scope_path)
        text = self.scope_path.read_text(encoding="utf-8").strip()
        if len(text) < 100:
            raise ValueError(
                f"Aims & Scope text looks too short ({len(text)} chars). "
                "Check that the file contains the thematic paragraphs."
            )
        log.info("Loaded Aims & Scope (%d characters).", len(text))
        return text

    
    ##### PRIVATE HELPERS #####

    @staticmethod
    def _check_exists(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Run fetcher.py first, or check the path."
            )

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

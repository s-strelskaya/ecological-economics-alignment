"""
computing thematic alignment scores, detects outliers,
aggregating drift statistics over time
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)


@dataclass
class OutlierReport:
    
    # container for outlier detection results

    # attributes
    """
    threshold: float
        score below which a paper is considered an outlier
    outliers: pd.DataFrame
        papers below the threshold, sorted by score ascending
    top_papers: pd.DataFrame
        top most aligned papers
    n_total: int
        total number of papers in the corpus
    """
    threshold: float
    outliers: pd.DataFrame
    top_papers: pd.DataFrame
    n_total: int

    @property
    def n_outliers(self) -> int:
        return len(self.outliers)

    @property
    def outlier_pct(self) -> float:
        return 100 * self.n_outliers / self.n_total

    def __str__(self) -> str:
        return (
            f"OutlierReport — threshold: {self.threshold:.4f} | "
            f"outliers: {self.n_outliers} ({self.outlier_pct:.1f}%)"
        )


class AlignmentScorer:
    
    ### computing cosine-similarity-based thematic alignment scores between
    # a corpus of paper abstracts and a reference document (Aims & Scope)

    # parameters
    """
    std_threshold: float
        number of standard deviations below the mean used to define
        outliers (default 1.5)
    top_n: int
        number of top-aligned papers to include in the outlier report
        (default 10)
    """

    # examples
    """
    scorer = AlignmentScorer()
    df = scorer.score(df, abstract_vecs, scope_vec)
    report = scorer.outlier_report(df)
    drift  = scorer.drift_by_year(df)
    """

    def __init__(
        self,
        std_threshold: float = 1.5,
        top_n: int = 10,
    ) -> None:
        self.std_threshold = std_threshold
        self.top_n         = top_n

    
    ##### PUBLIC INTERFACE #####

    def score(
        self,
        df: pd.DataFrame,
        abstract_vecs: np.ndarray,
        scope_vec: np.ndarray,
    ) -> pd.DataFrame:
        
        # adding an 'alignment_score' column to df

        ### cosine similarity between each abstract vector and the scope vector 
        # both are L2-normalised --> this reduces to
        #  a dot product: score = abstract_vec @ scope_vec

        # parameters
        """
        df: pd.DataFrame
            paper df from CorpusLoader
        abstract_vecs: np.ndarray
            shape (n_papers, dim): normalised abstract embeddings
        scope_vec: np.ndarray
            shape (dim,): normalised scope embedding
        """

        # returns
        """
        pd.DataFrame
            original df with new column 'alignment_score'
        """
        if len(df) != len(abstract_vecs):
            raise ValueError(
                f"DataFrame has {len(df)} rows but abstract_vecs has "
                f"{len(abstract_vecs)} rows. They must match"
            )

        scores = cosine_similarity(
            abstract_vecs,
            scope_vec.reshape(1, -1),
        ).flatten()

        df = df.copy()
        df["alignment_score"] = scores

        log.info(
            "Alignment scores computed. Mean: %.4f | Std: %.4f | "
            "Min: %.4f | Max: %.4f",
            scores.mean(), scores.std(), scores.min(), scores.max(),
        )
        return df

    def flag_editorial(self, df: pd.DataFrame) -> pd.DataFrame:
        
        ### adding 'is_editorial' boolean column flagging corrigenda,
        # errata, and book reviews -> content with no thematic substance

        # parameters
        """
        df: pd.DataFrame
            paper df with a 'title' column
        """

        # should return
        """
        pd.DataFrame
            original df with new column 'is_editorial'
        """
        keywords = ["corrigendum", "erratum", "book review", "correction to"]
        pattern = "|".join(keywords)
        df = df.copy()
        df["is_editorial"] = df["title"].str.lower().str.contains(
            pattern, na=False
        )
        n = df["is_editorial"].sum()
        log.info("Flagged %d editorial entries (corrigenda, errata, book reviews)", n)
        return df

    def outlier_report(self, df: pd.DataFrame) -> OutlierReport:
        
        ### identifying papers whose alignment score falls more than
        # std_threshold standard deviations below the mean

        # parameters
        """
        df: pd.DataFrame
            scored df (must contain 'alignment_score')
        """

        # should return
        """
        OutlierReport
            dataclass with threshold, outlier df, and top papers
        """
        self._require_column(df, "alignment_score")

        mean = df["alignment_score"].mean()
        std = df["alignment_score"].std()
        threshold = mean - self.std_threshold * std

        outliers = (
            df[df["alignment_score"] < threshold]
            .sort_values("alignment_score")
            .reset_index(drop=True)
        )
        top_papers = df.nlargest(self.top_n, "alignment_score").reset_index(drop=True)

        report = OutlierReport(
            threshold = threshold,
            outliers = outliers,
            top_papers = top_papers,
            n_total = len(df),
        )
        log.info(str(report))
        return report

    def drift_by_year(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # aggregating alignment scores by year to detect thematic drift

        # parameters
        """
        df: pd.DataFrame
            scored df (must contain 'alignment_score' and 'year')
        """

        # should return
        """
        pd.DataFrame
            one row per year with columns:
            year, mean, std, median, count, ci95
        """
        self._require_column(df, "alignment_score")
        self._require_column(df, "year")

        yearly = (
            df.groupby("year")["alignment_score"]
            .agg(mean="mean", std="std", median="median", count="count")
            .reset_index()
        )
        yearly["ci95"] = 1.96 * yearly["std"] / np.sqrt(yearly["count"])

        log.info(
            "Drift stats computed for %d years (%d–%d).",
            len(yearly), yearly["year"].min(), yearly["year"].max(),
        )
        return yearly

    def period_comparison(
        self,
        df: pd.DataFrame,
        split_year: int = 2018,
    ) -> pd.DataFrame:
        
        # comparing alignment scores between an early and a late period

        # parameters
        """
        df: pd.DataFrame
            scored df
        split_year: int
            papers up to and including this year form the early period
            (default 2018)
        """

        # should return
        """
        pd.DataFrame
            two-row df with mean, std, count per period
        """
        self._require_column(df, "alignment_score")
        self._require_column(df, "year")

        df = df.copy()
        df["period"] = df["year"].apply(
            lambda y: f"≤{split_year}" if y <= split_year else f">{split_year}"
        )
        return (
            df.groupby("period")["alignment_score"]
            .agg(mean="mean", std="std", count="count")
            .round(4)
        )

    
    ##### PRIVATE HELPERS #####

    @staticmethod
    def _require_column(df: pd.DataFrame, col: str) -> None:
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

"""Strategy drift detection — repetition, local optima, diversity collapse."""

import math
from collections import Counter
from dataclasses import dataclass

from autoresearch.results import ExperimentResult, classify_experiment


@dataclass
class StrategyAlert:
    """A single strategy drift alert."""
    level: str       # "info", "warning", "critical"
    category: str    # "repetition", "diversity", "similarity"
    message: str


@dataclass
class StrategyConfig:
    """Tunable thresholds for strategy drift detection."""
    repetition_threshold: int = 5       # same category N times without improvement
    diversity_window: int = 20          # recent experiments to check diversity
    diversity_min_entropy: float = 1.0  # Shannon entropy below this = convergence
    similarity_window: int = 10         # recent descriptions to check for duplicates
    similarity_overlap_threshold: float = 0.7  # keyword overlap > this = near-duplicate


class StrategyDrift:
    """Detects strategy-related drift in experiment proposals."""

    def __init__(self, config: StrategyConfig | None = None):
        self.config = config or StrategyConfig()

    def check(self, results: list[ExperimentResult]) -> list[StrategyAlert]:
        alerts = []
        alerts.extend(self._check_repetition(results))
        alerts.extend(self._check_diversity(results))
        alerts.extend(self._check_similarity(results))
        return alerts

    def _check_repetition(self, results: list[ExperimentResult]) -> list[StrategyAlert]:
        """Detect if the same category has been tried N times without improvement."""
        non_baseline = [r for r in results if r.status != "baseline"]
        if len(non_baseline) < self.config.repetition_threshold:
            return []

        # Check last N experiments for same category without a keep
        recent = non_baseline[-self.config.repetition_threshold:]
        categories = [classify_experiment(r.description) for r in recent]

        if len(set(categories)) == 1:
            # All same category
            any_kept = any(r.status == "keep" for r in recent)
            if not any_kept:
                return [StrategyAlert(
                    level="critical",
                    category="repetition",
                    message=f"Stuck in local optimum: last {self.config.repetition_threshold} "
                            f"experiments all in '{categories[0]}' category with no improvement. "
                            f"Switch to a different hyperparameter axis.",
                )]
        return []

    def _check_diversity(self, results: list[ExperimentResult]) -> list[StrategyAlert]:
        """Track Shannon entropy of category distribution in recent experiments."""
        non_baseline = [r for r in results if r.status != "baseline"]
        if len(non_baseline) < self.config.diversity_window:
            return []

        recent = non_baseline[-self.config.diversity_window:]
        categories = [classify_experiment(r.description) for r in recent]
        counts = Counter(categories)
        total = len(categories)

        # Shannon entropy
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        if entropy < self.config.diversity_min_entropy:
            dominant = counts.most_common(1)[0]
            return [StrategyAlert(
                level="warning",
                category="diversity",
                message=f"Strategy convergence: Shannon entropy = {entropy:.2f} "
                        f"(threshold: {self.config.diversity_min_entropy}). "
                        f"'{dominant[0]}' dominates with {dominant[1]}/{total} experiments. "
                        f"Explore underrepresented categories.",
            )]
        return []

    def _check_similarity(self, results: list[ExperimentResult]) -> list[StrategyAlert]:
        """Detect near-duplicate proposals via keyword overlap."""
        non_baseline = [r for r in results if r.status != "baseline"]
        if len(non_baseline) < 3:
            return []

        recent = non_baseline[-self.config.similarity_window:]
        latest_desc = recent[-1].description.lower()
        latest_words = set(latest_desc.split())

        if not latest_words:
            return []

        for r in recent[:-1]:
            other_words = set(r.description.lower().split())
            if not other_words:
                continue

            overlap = len(latest_words & other_words)
            union = len(latest_words | other_words)
            if union > 0:
                jaccard = overlap / union
                if jaccard > self.config.similarity_overlap_threshold:
                    return [StrategyAlert(
                        level="info",
                        category="similarity",
                        message=f"Near-duplicate proposal: '{recent[-1].description}' is "
                                f"{jaccard:.0%} similar to previous '{r.description}'. "
                                f"Consider a more differentiated approach.",
                    )]
        return []

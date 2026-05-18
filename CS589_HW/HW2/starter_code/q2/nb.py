import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from collections import Counter


class MultinomialNaiveBayes:
    def __init__(self, alpha: float = 0.0):
        self.alpha = alpha
        self.vocab: set[str] = set()
        self.log_pos_prior = 0.0
        self.log_neg_prior = 0.0
        self.pos_word_count: Counter[str] = Counter()
        self.neg_word_count: Counter[str] = Counter()
        self.pos_total: int = 0
        self.neg_total: int = 0

    def fit(
        self,
        positive_doc: list[list[str]],
        negative_doc: list[list[str]],
        vocab: set[str] | None = None,
        alpha: float | None = None,
    ):
        if alpha is not None:
            self.alpha = alpha

        if vocab is not None:
            self.vocab = vocab
        else:
            raise ValueError("vocab is none")

        log_pos_prior, log_neg_prior = self._fit_prior(positive_doc, negative_doc)

        self.log_pos_prior = log_pos_prior
        self.log_neg_prior = log_neg_prior

        self._count_word(positive_doc, negative_doc)

        return self

    def _fit_prior(self, positive_doc: list[list[str]], negative_doc: list[list[str]]):
        p = len(positive_doc)
        n = len(negative_doc)
        s = p + n
        pos_prior = p / s
        neg_prior = n / s
        return math.log(pos_prior), math.log(neg_prior)

    def _count_word(
        self, positive_doc: list[list[str]], negative_doc: list[list[str]]
    ) -> None:
        pos_counter: Counter[str] = Counter()
        neg_counter: Counter[str] = Counter()

        for doc in positive_doc:
            pos_counter.update(w for w in doc if w in self.vocab)  # 哎，想太多

        for doc in negative_doc:
            neg_counter.update(w for w in doc if w in self.vocab)

        self.pos_word_count = pos_counter
        self.neg_word_count = neg_counter
        self.pos_total = sum(pos_counter.values())
        self.neg_total = sum(neg_counter.values())

    def _p_w_pos(self, w: str) -> float:
        return math.log(
            (self.pos_word_count.get(w, 0) + self.alpha)
            / (self.pos_total + self.alpha * len(self.vocab))
        )

    def _p_w_neg(self, w: str) -> float:
        return math.log(
            (self.neg_word_count.get(w, 0) + self.alpha)
            / (self.neg_total + self.alpha * len(self.vocab))
        )

    def predict(self, doc: list[str]) -> float:
        log_pos_score = self.log_pos_prior
        log_neg_score = self.log_neg_prior

        for w in doc:
            if w not in self.vocab:
                continue
            log_pos_score += self._p_w_pos(w)
            log_neg_score += self._p_w_neg(w)

        return 1 if log_pos_score > log_neg_score else 0

    def evaluate(self, pos_test: list[list[str]], neg_test: list[list[str]]):
        TP = FP = TN = FN = 0

        for doc in pos_test:
            a = self.predict(doc)
            if a == 1:
                TP += 1
            else:
                FN += 1

        for doc in neg_test:
            b = self.predict(doc)
            if b == 1:
                FP += 1
            else:
                TN += 1

        total = TP + FP + TN + FN

        accuracy = (TP + TN) / total

        precision = TP / (TP + FP) if (TP + FP) > 0.0 else 0

        recall = TP / (TP + FN) if (TP + FN) > 0.0 else 0

        return {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }

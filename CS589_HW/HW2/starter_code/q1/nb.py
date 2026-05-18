import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


class MultinomialNaiveBayes:
    def __init__(self, alpha: float = 0.0):
        self.alpha = alpha
        self.vocab: set[str] = set()
        self.pos_prior = 0.0
        self.neg_prior = 0.0
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

        pos_prior, neg_prior = self._fit_prior(positive_doc, negative_doc)

        self.pos_prior = pos_prior
        self.neg_prior = neg_prior

        self._count_word(positive_doc, negative_doc)

        return self

    def _fit_prior(self, positive_doc: list[list[str]], negative_doc: list[list[str]]):
        p = len(positive_doc)
        n = len(negative_doc)
        s = p + n
        pos_prior = p / s
        neg_prior = n / s
        return pos_prior, neg_prior

    def _count_word(
        self, positive_doc: list[list[str]], negative_doc: list[list[str]]
    ) -> None:
        pos_counter: Counter[str] = Counter()
        neg_counter: Counter[str] = Counter()

        for doc in positive_doc:
            pos_counter.update(doc)

        for doc in negative_doc:
            neg_counter.update(doc)

        self.pos_word_count = pos_counter
        self.neg_word_count = neg_counter
        self.pos_total = sum(pos_counter.values())
        self.neg_total = sum(neg_counter.values())

    def _p_w_pos(self, w: str) -> float:
        return self.pos_word_count.get(w, 0) / self.pos_total

    def _p_w_neg(self, w: str) -> float:
        return self.neg_word_count.get(w, 0) / self.neg_total

    def predict(self, doc: list[str]) -> float:
        pos_score = self.pos_prior
        neg_score = self.neg_prior

        for w in doc:
            if w not in self.vocab:
                continue
            pos_score *= self._p_w_pos(w)
            neg_score *= self._p_w_neg(w)

        return 1 if pos_score > neg_score else 0

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

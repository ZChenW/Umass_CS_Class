# import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import operator
# import

# 写点注释，以防以后看不懂


class KNN:
    def __init__(self, k=3, weighted=False, eps=1e-6):
        self.k = k
        self.weighted = weighted
        self.eps = eps

    def fit(self, x, y):
        self.x = np.asarray(x)
        self.y = np.asarray(y)

    # 花费时间的写法，被_cal_with_matrix替代
    def _distance(self, v1, v2) -> float:
        return float(np.linalg.norm(v1 - v2))

    # Pick the most amnout labels
    def _vote(self, yk):
        vote_dict = {}
        for y in yk:
            if y not in vote_dict.keys():
                vote_dict[y] = 1
            else:
                vote_dict[y] += 1

        sort_vote_dict = sorted(
            vote_dict.items(), key=operator.itemgetter(1), reverse=True
        )
        return sort_vote_dict[0][0]

    def _weighted_vote(self, yk, dk):
        vote_dict = {}
        for y, d in zip(yk, dk):
            weight = 1.0 / (np.sqrt(d) + self.eps)
            if y not in vote_dict:
                vote_dict[y] = weight
            else:
                vote_dict[y] += weight
        sort_vote_dict = sorted(
            vote_dict.items(), key=operator.itemgetter(1), reverse=True
        )
        return sort_vote_dict[0][0]

    # 加速计算 ||A - B||_2^2 = A^2 + B^2 - 2*A@B.T
    def _cal_with_matrix(self, A, B):
        A_c = np.sum(A * A, axis=1, keepdims=True)  # [m, 1]
        B_c = np.sum(B * B, axis=1, keepdims=True).T  # [1, n]
        D = A_c + B_c - 2 * A @ B.T  # [m, n]
        return np.maximum(D, 0.0)

    # Training使用
    def train_predicate_leave_one_out(self):
        x = self.x
        D = self._cal_with_matrix(x, x)
        np.fill_diagonal(D, np.inf)  # leave itself out
        top_k = np.argsort(D, axis=1)[:, : self.k]
        y_pred = []
        for i in range(len(x)):
            neighbor_labels = self.y[top_k[i]]

            if self.weighted:
                neighbor_distances = D[i, top_k[i]]
                y_pred.append(self._weighted_vote(neighbor_labels, neighbor_distances))
            else:
                y_pred.append(self._vote(neighbor_labels))
        return np.asarray(y_pred)

    # Testing使用
    def predicate(self, x):
        D = self._cal_with_matrix(x, self.x)  # [m, n]
        top_k = np.argsort(D, axis=1)[:, : self.k]  # [m, k]
        y_pred = []
        for i in range(len(x)):
            neighbor_labels = self.y[top_k[i]]
            if self.weighted:
                neighbor_distances = D[i, top_k[i]]
                y_pred.append(self._weighted_vote(neighbor_labels, neighbor_distances))
            else:
                y_pred.append(self._vote(neighbor_labels))
        return np.asarray(y_pred)

    # 计算准确率，但是sklearn似乎也有一个函数计算，大意了
    def score(self, y_preds, y_true):
        if y_preds is None or y_true is None:
            y_preds = self.predicate(self.x)
            y_true = self.y
        y_preds = np.asarray(y_preds)
        y_true = np.asarray(y_true)
        return float(np.mean(y_preds == y_true))

import numpy as np
import pandas as pd
import operator


class DecisionTree:
    def __init__(self):
        pass

    ## Cal entropy
    def InformationEntropy(self, data):
        y_label = data.iloc[:, -1]
        info = 0
        enp = y_label.value_counts().values / len(y_label)
        info += -enp * (np.log2(enp))
        return np.sum(info)

    ## Cal Gain with attribute
    def InformationGain(self, data, a):
        Ent = self.InformationEntropy(data)
        choose_class = data[a].value_counts()
        gain = 0
        for i in choose_class.keys():
            w = choose_class[i] / data.shape[0]
            Env_v = self.InformationEntropy(data.loc[data[a] == i])
            gain += w * Env_v
        return Ent - gain

    def GetBestFeature(self, data):
        feature = data.columns[:-1]
        gain_list = {}
        for i in feature:
            gain = self.InformationGain(data, i)
            gain_list[i] = gain

        sort_gain_list = sorted(
            gain_list.items(), key=operator.itemgetter(1), reverse=True
        )
        return sort_gain_list[0][0]

    ## shan chu yijing shiyong guod ffeature
    def SpliteByFeature(self, data, bestfeature):
        attr = np.unique(data[bestfeature])
        new_data = [(a, data[data[bestfeature] == a]) for a in attr]
        update_new = [(i[0], i[1].drop([bestfeature], axis=1)) for i in new_data]
        return update_new

    def get_most_label(self, data):
        labels = data.iloc[:, -1]
        label_sort = labels.value_counts(sort=True)
        return label_sort.keys()[0]

    # Extra 2
    def getTree(self, data, threshold=False):
        ## 停止条件：1. attibute的标签全为一类 2. 样本只剩下一个特征
        lables = data.iloc[:, -1].value_counts()  # biaoqian

        if threshold:
            x = self.get_most_label(data)
            if lables[x] / len(data) >= 0.85:
                return x

        if len(lables) == 1 or data.shape[1] == 1:
            return lables.idxmax()

        bestfeature = self.GetBestFeature(data)
        tree = {bestfeature: {}}

        for a, dateframe in self.SpliteByFeature(data, bestfeature):
            tree[bestfeature][a] = self.getTree(dateframe, threshold)

        return tree

    def fit(self, train, threshold=False):
        self.tree = self.getTree(train, threshold)

    def predit_one(self, a):
        # self.miss = 0
        node = self.tree
        while isinstance(node, dict):
            feature = next(iter(node))
            branches = node[feature]
            value = a[feature]
            if value not in branches:  ##baozheng bu chuxian keyerror
                # self.miss += 1
                node = next(iter(branches.values()))
            else:
                node = branches[value]
        return node

    def predit(self, dp):  # dp => dataframe
        preds = []
        for _, i in dp.iterrows():
            preds.append(self.predit_one(i))
        # print(self.miss)
        return preds

    def score(self, dp):
        labels = dp.columns[-1]
        X = dp.drop(columns=[labels])
        y = dp[labels].to_numpy()
        y_pred = np.array(self.predit(X))
        return np.mean(y_pred == y)

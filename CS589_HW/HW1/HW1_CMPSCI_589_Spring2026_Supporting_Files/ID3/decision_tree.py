import numpy as np
import pandas as pd


class DecisionTree:
    def __init__(self, max_lenght=None, min_samples_split=2, min_gain=1e-6, node_chose_limit=None, random_seed=None):
        self.max_lenght = max_lenght 
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain 
        self.node_chose_limit = node_chose_limit 
        self.random_seed = np.random.RandomState(random_seed)
        self.tree = None 

    ## Cal entropy
    def InformationEntropy(self, data):
        y_label = data.iloc[:, -1]
        info = 0
        enp = y_label.value_counts().values / len(y_label)
        info += -enp * (np.log2(enp))
        return np.sum(info)

    ## Cal Gain
    def is_numeric(self, data):
        return pd.api.types.is_numeric_dtype(data)
    
    def NumericInformationGain(self, data, a):
        Ent = self.InformationEntropy(data)
        threshold = data[a].mean()
        left = data[data[a] <= threshold]
        right = data[data[a] > threshold]
        if len(left) == 0 or len(right) == 0:
            return 0.0, threshold
        left_ratio = len(left) / len(data)
        right_ratio = len(right) / len(data)
        NewEnt = left_ratio * self.InformationEntropy(left) + right_ratio * self.InformationEntropy(right)
        gain = Ent - NewEnt
        return gain, threshold

     ## Cal Gain with attribute(numeric attribute)
    def InformationGain(self, data, a):
        if self.is_numeric(data[a]):
            gain, threshold = self.NumericInformationGain(data, a)
            return {"gain": gain,"feature": a, "threshold": threshold, "is_numeric": True}
        else:
            Ent = self.InformationEntropy(data)
            choose_class = data[a].value_counts()
            gain = 0
            for i in choose_class.keys():
                w = choose_class[i] / data.shape[0]
                Env_v = self.InformationEntropy(data.loc[data[a] == i])
                gain += w * Env_v
            return {"gain": Ent - gain, "feature": a, "threshold": None, "is_numeric": False}

    # hhh，怎么这么像vote
    def GetBestFeature(self, data):
        feature = list(data.columns[:-1])
        if self.node_chose_limit is not None and len(feature) > self.node_chose_limit:
            feature = self.random_seed.choice(feature, self.node_chose_limit, replace=False)
        best_splite = None
        for i in feature:
            result = self.InformationGain(data, i)
            if best_splite is None or result["gain"] > best_splite["gain"]:
                best_splite = result

        return best_splite

    ## shan chu yijing shiyong guod ffeature
    def SpliteByFeatureCate(self, data, bestfeature):
        attr = np.unique(data[bestfeature])
        new_data = [(a, data[data[bestfeature] == a]) for a in attr]
        update_new = [(i[0], i[1].drop([bestfeature], axis=1)) for i in new_data]
        return update_new

    def SpliteByFeatureNum(self, data, bestfeature, threshold):
        left = data[data[bestfeature] <= threshold] # We don not need drop
        right = data[data[bestfeature] > threshold]
        return {"leq": left, "gt": right}

    def get_most_label(self, data):
        labels = data.iloc[:, -1]
        label_sort = labels.value_counts(sort=True)
        return label_sort.keys()[0]

    # Extra 2
    def getTree(self, data, depth=0):
        ## 停止条件：1. attibute的标签全为一类 2. 样本只剩下一个特征
        lables = data.iloc[:, -1]  # biaoqian
        x = lables.value_counts()

        if lables.nunique() == 1:
            return {"type": "leaf", "prediction": lables.iloc[0]}

        if data.shape[1] == 1:
            return {"type": "leaf", "prediction": x.idxmax()}
        
        if self.max_lenght is not None and depth >= self.max_lenght:
            return {"type": "leaf", "prediction": x.idxmax()}

        if len(data) < self.min_samples_split:
            return {"type": "leaf", "prediction": x.idxmax()}

        bestfeature = self.GetBestFeature(data)

        if bestfeature["gain"] < self.min_gain:
            return {"type": "leaf", "prediction": x.idxmax()}
        
        node = {
        "type": "node",
        "prediction": x.idxmax(),
        "feature": bestfeature["feature"],
        "is_numeric": bestfeature["is_numeric"],
        "threshold": bestfeature["threshold"],
        "children": {},
        }

        if bestfeature["is_numeric"]:
            threshold = bestfeature["threshold"]
            splite = self.SpliteByFeatureNum(data, bestfeature["feature"], threshold)

            if len(splite["leq"]) == 0 or len(splite["gt"]) == 0:
                return {"type": "leaf", "prediction": x.idxmax()}

            node["children"]["leq"] = self.getTree(splite["leq"], depth=depth+1)
            node["children"]["gt"] = self.getTree(splite["gt"], depth=depth+1)
        else:
            for a, dateframe in self.SpliteByFeatureCate(data, bestfeature["feature"]):
                node["children"][a] = self.getTree(dateframe, depth=depth+1)

        return node

    def fit(self, train):
        self.tree = self.getTree(train, depth=0)

    def predit_one(self, a):
        # self.miss = 0
        node = self.tree
        while node["type"] != "leaf":
            feature = node["feature"]
            if node["is_numeric"]:
                threshold = node["threshold"]
                if a[feature] <= threshold:
                    next_node = node["children"]["leq"]
                else:
                    next_node = node["children"]["gt"]
                
                if next_node is None:
                    return node["prediction"]
                
                node = next_node

            else:
                value = a[feature]
                if value not in node["children"]:  ##baozheng bu chuxian keyerror
                # self.miss += 1
                    return node["prediction"]
                else:
                    node = node["children"][value]
        return node["prediction"]

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

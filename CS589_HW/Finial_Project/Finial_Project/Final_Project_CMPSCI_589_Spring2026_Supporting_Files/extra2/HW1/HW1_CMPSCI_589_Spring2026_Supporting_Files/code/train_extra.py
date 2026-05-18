import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from knn import KNN


def random_seed(seed=42):
    np.random.seed(seed)


def read_csv(file_path, label_name=None):
    df = pd.read_csv(file_path)
    if label_name is not None:
        df = df[[i for i in df.columns if i != label_name] + [label_name]]
    return df


def cal_acc(y_true: np.ndarray, y_pred: np.ndarray):
    return np.mean(y_true == y_pred)


def cal_conf_matrix(y_true, y_pred, label):
    tp = fp = tn = fn = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == label and yp == label:
            tp += 1
        elif yt != label and yp == label:
            fp += 1
        elif yt != label and yp != label:
            tn += 1
        elif yt == label and yp != label:
            fn += 1

    return tp, fp, tn, fn


def cal_precision(y_true, y_pred, label):
    tp, fp, _, _ = cal_conf_matrix(y_true, y_pred, label)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def cal_recall(y_true, y_pred, label):
    tp, _, _, fn = cal_conf_matrix(y_true, y_pred, label)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def cal_f1_score(y_true, y_pred, label):
    prec = cal_precision(y_true, y_pred, label)
    rec = cal_recall(y_true, y_pred, label)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0


def cal_f1_macro(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(y_true)
    f1_scores = []
    for label in labels:
        f1_scores.append(cal_f1_score(y_true, y_pred, label))
    return np.mean(f1_scores)


def normalize(X):
    max_value = X.max(axis=0)
    min_value = X.min(axis=0)
    return max_value, min_value


def normalize_dataset(X, max_value, min_value):
    return (X - min_value) / (max_value - min_value + 1e-8)


def split(df, label_name):
    X = df.drop(columns=[label_name]).values.astype(float)
    y = df[label_name].values
    return X, y


def k_fold(df, label_name, k=10, random_seed=42):
    random = np.random.RandomState(random_seed)
    folds = [[] for _ in range(k)]

    # 获得分组后的index
    all_labed = {}
    for idx, label_value in enumerate(df[label_name].values):
        if label_value not in all_labed:
            all_labed[label_value] = []
        all_labed[label_value].append(idx)

    for _, group_idxs in all_labed.items():
        group_idxs = group_idxs.copy()
        random.shuffle(group_idxs)

        for i, idx in enumerate(group_idxs):
            folds[i % k].append(idx)  # 保持比例相同

    # 转化成dataFrame, reset index
    fold = []
    for i in range(k):
        fold.append(df.iloc[folds[i]].reset_index(drop=True))

    return fold


def train_test(fold, text_fold_idx):
    test_fold = fold[text_fold_idx].reset_index(drop=True)
    train = [fold[i] for i in range(len(fold)) if i != text_fold_idx]
    train_fold = pd.concat(train, axis=0).reset_index(drop=True)
    return train_fold, test_fold


def prepossing(df, label_name):
    df = df.copy()
    cate_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cate_cols = [c for c in cate_cols if c != label_name]
    df = pd.get_dummies(df, columns=cate_cols, dtype=float)
    return df


# def __init__(self, n_trees=10, max_lenght=None, min_samples_split=2,
# min_gain=1e-6, node_chose_limit=None, random_seed=None):
def cal_k(df, label_name, k=10, random_seed=42, label=1, kp=None):

    folds = k_fold(df, label_name, k, random_seed)

    accs = []
    prec = []
    recall = []
    f1 = []

    for i in range(k):
        train_df, test_df = train_test(folds, i)

        X_train, y_tain = split(train_df, label_name)
        X_test, y_test = split(test_df, label_name)

        max_val, min_val = normalize(X_train)
        X_train = normalize_dataset(X_train, max_val, min_val)
        X_test = normalize_dataset(X_test, max_val, min_val)

        knn = KNN(kp)
        knn.fit(X_train, y_tain)
        y_pred = knn.predicate(X_test)

        accs.append(cal_acc(y_test, y_pred))
        prec.append(cal_precision(y_test, y_pred, label))
        recall.append(cal_recall(y_test, y_pred, label))
        f1.append(cal_f1_macro(y_test, y_pred))

    return {
        "k_neighbors": kp,
        "acc": np.mean(accs),
        "prec": np.mean(prec),
        "recall": np.mean(recall),
        "f1": np.mean(f1),
    }


def make_graph_plt(df, x_lab, y_lab, title, dir, filename=None):
    plt.figure()
    plt.plot(df[x_lab], df[y_lab], marker="o")
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, filename), dpi=200, bbox_inches="tight")
    plt.close()


def cal_one_dataset(
    dataset_name,
    csv_path,
    label_col,
    dir,
    filename=None,
    ks=None,
    k=10,
    random_seed=42,
    label=None,
):
    df = read_csv(csv_path, label_name="stress_level")
    df = prepossing(df, label_name="stress_level")

    results = []

    for i in ks:
        print(f"kp: {i}")

        result = cal_k(
            df=df,
            label_name=label_col,
            k=k,
            random_seed=random_seed,
            label=label,
            kp=i,
        )
        results.append(result)

    result_df = pd.DataFrame(results)
    os.makedirs(dir, exist_ok=True)
    result_df.to_csv(os.path.join(dir, f"{dataset_name}_results.csv"), index=False)

    make_graph_plt(
        result_df,
        x_lab="k_neighbors",
        y_lab="acc",
        title=f"{dataset_name} KNN Accuracy vs k",
        dir=dir,
        filename=f"{dataset_name}_knn_acc.png",
    )

    make_graph_plt(
        result_df,
        x_lab="k_neighbors",
        y_lab="f1",
        title=f"{dataset_name} KNN F1 Score vs k",
        dir=dir,
        filename=f"{dataset_name}_knn_f1.png",
    )

    return result_df


def x(k=10, kp=None):
    random_seed(42)

    spam_csv_path = "../datasets/reels_attention_span_dataset_12000.csv"
    spam_csv_label_name = "stress_level"

    # def cal_one_dataset(dataset_name, csv_path, label_col, dir,
    # filename=None, ntree_values=(1, 5, 10, 20, 30, 40, 50), k=5,
    # random_seed=42, max_lenght=None, min_samples_split=2,
    # min_gain=1e-6, node_chose_limit=None, label=1

    spam_result = cal_one_dataset(
        dataset_name=f"spam_{k}",
        csv_path=spam_csv_path,
        label_col=spam_csv_label_name,
        dir=f"../picture/spam_k{k}",
        ks=kp,
        k=k,
        random_seed=42,
        label=1,
    )

    print(f"k = {k},Mnist Result:")
    print(spam_result)


if __name__ == "__main__":
    ks = [1, 3, 5, 7, 9, 11]
    x(k=10, kp=ks)

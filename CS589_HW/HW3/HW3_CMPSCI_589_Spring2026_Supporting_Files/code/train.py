import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from random_forest import RandomForest

def random_seed(seed=42):
    np.random.seed(seed)

def read_csv(file_path):
    return pd.read_csv(file_path)

def cal_acc(y_true : np.ndarray, y_pred: np.ndarray):
    return np.mean(y_true == y_pred)

def cal_conf_matrix(y_true, y_pred, positive_label=1):
    tp = fp = tn = fn = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == positive_label and yp == positive_label:
            tp += 1
        elif yt != positive_label and yp == positive_label:
            fp += 1
        elif yt != positive_label and yp != positive_label:
            tn += 1
        elif yt == positive_label and yp != positive_label:
            fn += 1

    return tp, fp, tn, fn

def cal_precision(y_true, y_pred, positive_label=1):
    tp, fp, _, _ = cal_conf_matrix(y_true, y_pred, positive_label)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def cal_recall(y_true, y_pred, positive_label=1):
    tp, _, _, fn = cal_conf_matrix(y_true, y_pred, positive_label)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def cal_f1_score(y_true, y_pred, positive_label=1):
    prec = cal_precision(y_true, y_pred, positive_label)
    rec = cal_recall(y_true, y_pred, positive_label)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

def k_fold(df, label_name, k=5, random_seed=42):
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
            folds[i%k].append(idx) # 保持比例相同

    #转化成dataFrame, reset index
    fold = []
    for i in range(k):
        fold.append(df.iloc[folds[i]].reset_index(drop=True))

    return fold

def train_test(fold, text_fold_idx):
    test_fold = fold[text_fold_idx].reset_index(drop=True)
    train = [fold[i] for i in range(len(fold)) if i != text_fold_idx]
    train_fold = pd.concat(train, axis=0).reset_index(drop=True)
    return train_fold, test_fold

# def __init__(self, n_trees=10, max_lenght=None, min_samples_split=2, 
# min_gain=1e-6, node_chose_limit=None, random_seed=None):
def cal_ntrees(df, label_name, k=5, random_seed=42, positive_label=1,
            n_trees=10, max_lenght=None, min_samples_split=2,
            min_gain=1e-6, node_chose_limit=None):

            folds = k_fold(df, label_name, k, random_seed)

            accs = [] 
            prec = [] 
            recall = [] 
            f1 = []

            for i in range(k):
                train, test = train_test(folds, i)

                random_forest = RandomForest(
                    n_trees=n_trees,
                    max_lenght=max_lenght,
                    min_samples_split=min_samples_split,
                    min_gain=min_gain,
                    node_chose_limit=node_chose_limit,
                    random_seed=random_seed
                )

                random_forest.fit(train)
                X_test = test.drop(columns=[label_name])
                y_test = test[label_name]
                y_pred = random_forest.predict(X_test)

                accs.append(cal_acc(y_test, y_pred))
                prec.append(cal_precision(y_test, y_pred, positive_label))
                recall.append(cal_recall(y_test, y_pred, positive_label))
                f1.append(cal_f1_score(y_test, y_pred, positive_label))


            return {
                "ntrees": n_trees,
                "acc": np.mean(accs),
                "prec": np.mean(prec),
                "recall": np.mean(recall),
                "f1": np.mean(f1)
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
    ntree_values=(1, 5, 10, 20, 30, 40, 50),
    k=5,
    random_seed=42,
    max_lenght=None,
    min_samples_split=2,
    min_gain=1e-6,
    node_chose_limit=None,
    positive_label=1
):
    df = read_csv(csv_path)

    results = []

    for i in ntree_values:
        print(f"Ntree: {i}")

        result = cal_ntrees(
            df=df,
            label_name=label_col,
            k=k,
            random_seed=random_seed,
            positive_label=positive_label,
            n_trees=i,
            max_lenght=max_lenght,
            min_samples_split=min_samples_split,
            min_gain=min_gain,
            node_chose_limit=node_chose_limit
        )
        results.append(result)

    result_df = pd.DataFrame(results)
    os.makedirs(dir, exist_ok=True)
    result_df.to_csv(os.path.join(dir, f"{dataset_name}_results.csv"), index=False)

    make_graph_plt(
        result_df,
        x_lab="ntrees",
        y_lab="acc",
        title=f"{dataset_name} Accuracy vs Ntrees",
        dir=dir,
        filename = f"{dataset_name}_acc.png",
    )

    make_graph_plt(
        result_df,
        x_lab="ntrees",
        y_lab="prec",
        title=f"{dataset_name} Precision vs Ntrees",
        dir=dir,
        filename = f"{dataset_name}_prec.png",
    )

    make_graph_plt(
        result_df,
        x_lab="ntrees",
        y_lab="recall",
        title=f"{dataset_name} Recall vs Ntrees",
        dir=dir,
        filename = f"{dataset_name}_recall.png"
    )

    make_graph_plt(
        result_df,
        x_lab="ntrees",
        y_lab="f1",
        title=f"{dataset_name} F1 Score vs Ntrees",
        dir=dir,
        filename = f"{dataset_name}_f1.png"
    )

    return result_df

def x(k=5, max_length=10, min_samples_split=2):
    random_seed(42)

    wdbc_csv_path = "/home/chakew/Downloads/spring_2026/cs589/HW3/HW3_CMPSCI_589_Spring2026_Supporting_Files/dataset/wdbc.csv"
    wdbc_csv_label_name = "label"

    loan_csv_path = "/home/chakew/Downloads/spring_2026/cs589/HW3/HW3_CMPSCI_589_Spring2026_Supporting_Files/dataset/loan.csv"
    loan_csv_label_name = "label"

    # def cal_one_dataset(dataset_name, csv_path, label_col, dir,
    # filename=None, ntree_values=(1, 5, 10, 20, 30, 40, 50), k=5,
    # random_seed=42, max_lenght=None, min_samples_split=2,
    # min_gain=1e-6, node_chose_limit=None, positive_label=1

    wdbc_result = cal_one_dataset(
        dataset_name=f"WDBC_{k}_depth_{max_length}_minsplit{min_samples_split}",
        csv_path=wdbc_csv_path,
        label_col=wdbc_csv_label_name,
        dir=f"../picture/wdbc_k{k}_depth{max_length}_minsplit{min_samples_split}",
        ntree_values=(1, 5, 10, 20, 30, 40, 50),
        k=k,
        random_seed=42,
        max_lenght=max_length,
        min_samples_split=min_samples_split,
        min_gain=1e-6,
        node_chose_limit=5, # [\sqrt(30)] = 5
        positive_label=1
    )
    
    loan_result = cal_one_dataset(
        dataset_name=f"Loan_{k}_depth_{max_length}_minsplit{min_samples_split}",
        csv_path=loan_csv_path,
        label_col=loan_csv_label_name,
        dir=f"../picture/loan_k{k}_depth{max_length}_minsplit{min_samples_split}",
        ntree_values=(1, 5, 10, 20, 30, 40, 50),
        k=k,
        random_seed=42,
        max_lenght=max_length,
        min_samples_split=min_samples_split,
        min_gain=1e-6,
        node_chose_limit=3, # [\sqrt(11)] = 3 
        positive_label=1
    )

    print(f"k = {k}, max_depth = {max_length}, min_samples_split = {min_samples_split} WDBC Result:")
    print(wdbc_result)
    print(f"\nk = {k}, max_depth = {max_length}, min_samples_split = {min_samples_split} Loan Result:")
    print(loan_result)


if __name__ == "__main__":
    max_depths = [5, 8, 10, 15]
    for i in max_depths:
        print(f"Max Depth: {i}, K: 5, Min Samples Split: 2")
        x(k=5, max_length=i)

    min_splits = [5, 10]
    for m in min_splits:
        print(f"Max Depth: 10, K: 5, Min Samples Split: {m}")
        x(k=5, max_length=10, min_samples_split=m)
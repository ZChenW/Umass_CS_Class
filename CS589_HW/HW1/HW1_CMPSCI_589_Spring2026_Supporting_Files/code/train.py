from knn import KNN
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


ks = [i for i in range(52) if i % 2 == 1]  # 1~51 的奇数


def read_csv_file(csv_path):
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] != 31:
        raise ValueError("Incorrect csv formate")
    X = df.iloc[:, :-1].to_numpy(dtype=np.float32)  # 569*30
    y = df.iloc[:, -1].to_numpy(dtype=np.float32)  # 569*1

    assert X.shape[1] == 30  # column
    assert y.shape[0] == X.shape[0]  # Row

    return X, y


# 拿到 acc for train and test
def prepare(csv_path, k, seed, normalize=True):
    X, y = read_csv_file(csv_path=csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=seed
    )
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    module = KNN(k=k)
    module.fit(X_train, y_train)

    y_test_p = module.predicate(X_test)
    y_train_p = module.train_predicate_leave_one_out()

    test_acc = module.score(y_test_p, y_test)
    train_acc = module.score(y_train_p, y_train)

    return test_acc, train_acc


# 一轮训练
def train(csv_path, normalize=True):
    base_seed = 0
    train_acc_mean = []
    train_acc_std = []
    test_acc_mean = []
    test_acc_std = []
    for i in ks:
        train_acc_list = []
        test_acc_list = []
        for r in range(20):
            seed = base_seed + 100 * r
            test_acc, train_acc = prepare(csv_path, i, seed, normalize=normalize)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(f"k={i}, repeat={r}", flush=True)
        train_acc_mean.append(np.mean(train_acc_list))
        test_acc_mean.append(np.mean(test_acc_list))
        train_acc_std.append(np.std(train_acc_list, ddof=1))
        test_acc_std.append(np.std(test_acc_list, ddof=1))

    return train_acc_mean, test_acc_mean, train_acc_std, test_acc_std


# 低效，但是懒得改了
def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前文件位置
    csv_path = os.path.join(BASE_DIR, "..", "datasets", "wdbc.csv")
    # print("BASE_DIR =", BASE_DIR)
    # print("csv_path =", csv_path)
    # print("exists?  =", os.path.exists(csv_path))
    tr_mean, te_mean, tr_std, te_std = train(csv_path=csv_path, normalize=True)
    _, te_mean_no, _, te_std_no = train(csv_path=csv_path, normalize=False)

    out_dir = os.path.join(BASE_DIR, "pictures")
    os.makedirs(out_dir, exist_ok=True)

    Figure1 = plt.figure()
    plt.errorbar(ks, tr_mean, yerr=tr_std, capsize=3)
    plt.xlabel("Value of k")
    plt.ylabel("Accuracy")
    plt.title("Train set Accuracy vs k (normalize = true)")
    plt.grid()
    plt.ylim(0.7, 1)
    plt.savefig(
        os.path.join(out_dir, "q1_1_train_normalized.png"), dpi=200, bbox_inches="tight"
    )
    plt.close()

    Figure2 = plt.figure()
    plt.errorbar(ks, te_mean, yerr=te_std, capsize=3)
    plt.xlabel("Value of k")
    plt.ylabel("Accuracy")
    plt.title("Testset Accuracy vs k (normalize = true)")
    plt.ylim(0.7, 1)
    plt.grid()
    plt.savefig(
        os.path.join(out_dir, "q1_2_test_normalized.png"), dpi=200, bbox_inches="tight"
    )
    plt.close()

    Figure3 = plt.figure()
    plt.errorbar(ks, te_mean_no, yerr=te_std_no, fmt="o-", capsize=3)
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.title("Test Accuracy vs k (normalized = false)")
    plt.grid()
    plt.ylim(0.7, 1)
    plt.savefig(
        os.path.join(out_dir, "q1_6_test_not_normalized.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()


# print(df.shape)
# print(df.head())

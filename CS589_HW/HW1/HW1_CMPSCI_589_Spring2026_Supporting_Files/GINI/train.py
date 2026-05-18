from decision_tree import DecisionTree
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def read_csv_file(csv_path):
    df = pd.read_csv(csv_path)
    return df


def train_test_split(dp, test_ration=0.2, seed=None):
    dp_shuffled = dp.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_number = int(round(len(dp_shuffled) * test_ration))
    test_df = dp_shuffled.iloc[:n_number].reset_index(drop=True)
    train_df = dp_shuffled.iloc[n_number:].reset_index(drop=True)
    return test_df, train_df


def prepare(sp, n=100, test_ration=0.2):
    train_acc = []
    test_acc = []

    for i in range(n):
        test_df, train_df = train_test_split(dp=sp, test_ration=test_ration, seed=i)
        module = DecisionTree()
        module.fit(train_df)

        train_acc.append(module.score(train_df))
        test_acc.append(module.score(test_df))

        print(f"K = {i}", flush=True)

    train_acc = np.array(train_acc)
    test_acc = np.array(test_acc)
    return train_acc, test_acc


def make_graph_plt(acc, title, dir, filename=None, name=None):
    plt.figure()
    plt.hist(acc, bins=15)
    plt.xlabel("Accuracy")
    plt.ylabel(f"Accuracy Frequency on {name} Data")
    plt.title(title)
    plt.grid()
    #    plt.ylim(0, 1)
    plt.savefig(os.path.join(dir, filename), dpi=200, bbox_inches="tight")
    plt.close()


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前文件位置
    csv_path = os.path.join(BASE_DIR, "..", "datasets", "car.csv")

    out_dir = os.path.join(BASE_DIR, "pictures")
    os.makedirs(out_dir, exist_ok=True)

    train_acc, test_acc = prepare(read_csv_file(csv_path), n=100)

    make_graph_plt(
        train_acc, "Train Accuracy", out_dir, filename="train_hist.png", name="train"
    )

    make_graph_plt(
        test_acc, "Test Accuracy", out_dir, filename="test_hist.png", name="test"
    )

    print(f"Train mean={train_acc.mean():.4f}, std={train_acc.std(ddof=0):.4f}")
    print(f"Test  mean={test_acc.mean():.4f}, std={test_acc.std(ddof=0):.4f}")


if __name__ == "__main__":
    main()

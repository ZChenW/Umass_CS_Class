import os
import numpy as np
from feedforward import Feedforward
import matplotlib.pyplot as plt
import pandas as pd


def random_seed(seed):
    np.random.seed(seed)


def read_csv(file_path, label_name=None):
    df = pd.read_csv(file_path)
    if label_name is not None:
        df = df[[i for i in df.columns if i != label_name] + [label_name]]
    return df


def make_graph_plt(df, x_lab, y_lab, title, out_dir, filename=None):
    if filename is None:
        filename = "plot.png"
    plt.figure()
    plt.plot(df[x_lab], df[y_lab], marker="o")
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200, bbox_inches="tight")
    plt.close()


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


def cal_acc(y_true: np.ndarray, y_pred: np.ndarray):
    return np.mean(y_true == y_pred)


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


def k_fold(df, label_name, k=10, random_seed=42):
    ran = np.random.RandomState(random_seed)
    folds = [[] for _ in range(k)]

    # 获得分组后的index
    all_labed = {}
    for idx, label_value in enumerate(df[label_name].values):
        if label_value not in all_labed:
            all_labed[label_value] = []
        all_labed[label_value].append(idx)

    for _, group_idxs in all_labed.items():
        group_idxs = group_idxs.copy()
        ran.shuffle(group_idxs)

        for i, idx in enumerate(group_idxs):
            folds[i % k].append(idx)  # 保持比例相同

    # 转化成dataFrame, reset index
    fold = []
    for i in range(k):
        fold.append(df.iloc[folds[i]].reset_index(drop=True))

    return fold


def predict(model, X, threshold=0.5):
    y_pred, _, _ = model.forward(X)
    return (y_pred >= threshold).astype(int)


def split(df, label_name="label"):
    X = df.drop(columns=[label_name]).values.astype(float)
    y = df[label_name].values.reshape(-1, 1).astype(float)
    return X, y


def normalize(X):
    max_value = X.max(axis=0)
    min_value = X.min(axis=0)
    return max_value, min_value


def normalize_dataset(X, max_value, min_value):
    return (X - min_value) / (max_value - min_value + 1e-8)


def train_test(fold, test_fold_idx):
    test_fold = fold[test_fold_idx].reset_index(drop=True)
    train = [fold[i] for i in range(len(fold)) if i != test_fold_idx]
    train_fold = pd.concat(train, axis=0).reset_index(drop=True)
    return train_fold, test_fold


def train(
    model, X_train, Y_train, X_val, Y_val, epochs=1000, learning_rate=0.01, lambda_=0.0
):
    history = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        y_pred, _, acts = model.forward(X_train)
        train_loss = model.compute_loss(y_pred, Y_train, lambda_)

        dw, db = model.backward(Y_train, y_pred, acts, lambda_)

        model.step(dw, db, learning_rate)

        y_pred_val, _, _ = model.forward(X_val)
        val_loss = model.compute_loss(y_pred_val, Y_val, lambda_)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 1000 == 0 or epoch == 1:
            print(
                f"Epoch {epoch}/{epochs} || Train Loss: {train_loss:.4f} || Val Loss: {val_loss:.4f}"
            )

    history_df = pd.DataFrame(history)
    return model, history_df


def train_test_on_fold(
    df,
    label_name="label",
    arch_linux=[],
    lambdas=[],
    k=10,
    random_seed=42,
    positive_label=1,
    epochs=1000,
    learning_rate=0.05,
):

    folds = k_fold(df, label_name, k, random_seed)
    result = []

    for arch in arch_linux:
        for mint in lambdas:
            fold_acc = []
            fold_f1 = []

            for i in range(k):
                print(f"\nFold {i + 1}/{k} || arch={arch} || lambda={mint}")
                train_df, test_df = train_test(folds, i)

                X_train, y_tain = split(train_df)
                X_test, y_test = split(test_df)

                print("X_train:", X_train.shape, "y_train:", y_tain.shape)
                print("X_test:", X_test.shape, "y_test:", y_test.shape)

                max_val, min_val = normalize(X_train)
                X_train = normalize_dataset(X_train, max_val, min_val)
                X_test = normalize_dataset(X_test, max_val, min_val)

                model = Feedforward(arch)

                model, _ = train(
                    model,
                    X_train,
                    y_tain,
                    X_test,
                    y_test,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    lambda_=mint,
                )

                y_pred = predict(model, X_test)
                acc = cal_acc(y_test.flatten(), y_pred.flatten())
                f1 = cal_f1_score(y_test.flatten(), y_pred.flatten())

                fold_acc.append(acc)
                fold_f1.append(f1)

            result.append(
                {
                    "arch": arch,
                    "lambda": mint,
                    "acc": np.mean(fold_acc),
                    "f1": np.mean(fold_f1),
                }
            )
    return pd.DataFrame(result)


def prepossing(df, label_name="label"):
    df = df.copy()

    cate_cols = [c for c in df.columns if c.endswith("cat") and c != label_name]

    df = pd.get_dummies(df, columns=cate_cols, dtype=float)

    return df


def learning_curve(
    df,
    label_name="label",
    best_arch=None,
    best_lambda=None,
    k=10,
    random_seed=42,
    positive_label=1,
    epochs=1000,
    learning_rate=0.05,
    train_size=None,
):
    if train_size is None:
        train_size = [5, 10, 20, 40, 80, 160, 320]

    folds = k_fold(df, label_name, k, random_seed)
    result = {"train_size": [], "test_loss": []}

    train_df, test_df = train_test(folds, 1)

    X_train, y_train = split(train_df, label_name)
    X_test, y_test = split(test_df, label_name)

    max_val, min_val = normalize(X_train)

    X_train = normalize_dataset(X_train, max_val, min_val)
    X_test = normalize_dataset(X_test, max_val, min_val)

    for n in train_size:
        print(f"\nTrain_size {n} || arch={best_arch} || lambda={best_lambda}")

        if n > len(X_train):
            break

        sub_X_train = X_train[:n]
        sub_y_train = y_train[:n]

        model = Feedforward(best_arch)

        model, _ = train(
            model,
            sub_X_train,
            sub_y_train,
            X_test,
            y_test,
            epochs=epochs,
            learning_rate=learning_rate,
            lambda_=best_lambda,
        )

        y_pred, _, _ = model.forward(X_test)
        loss = model.compute_loss(y_pred, y_test, best_lambda)

        result["train_size"].append(n)
        result["test_loss"].append(loss)

    return pd.DataFrame(result)


if __name__ == "__main__":
    random_seed(42)

    """
    wdbc_csv_path = "../datasets/wdbc.csv"
    wdbc_csv_label_name = "label"

    loan_csv_path = "../datasets/loan.csv"
    loan_csv_label_name = "label"

    #################_WDBC_#################
    wdbc_df = read_csv(wdbc_csv_path)
    dir = "../result/"
    os.makedirs(dir, exist_ok=True)
    credit_input_dim = wdbc_df.shape[1] - 1
    """

    """
    wdbc_result = train_test_on_fold(
        wdbc_df,
        label_name=wdbc_csv_label_name,
        arch_linux=[
            [credit_input_dim, 2, 1],
            [credit_input_dim, 4, 1],
            [credit_input_dim, 8, 1],
            [credit_input_dim, 16, 1],
            [credit_input_dim, 8, 4, 1],
            [credit_input_dim, 16, 8, 4, 1], 
            [credit_input_dim, 16, 8, 4, 2, 1],
        ],
        lambdas=[0.01, 0.05, 0.1, 0.3],
        k=5,
        random_seed=42,
        positive_label=1,
        epochs=10000,
        learning_rate=0.05,
    )

    wdbc_result.to_csv(os.path.join(dir, "wdbc_result.csv"), index=False)

    print(wdbc_result) 

    wdbc_best_result = learning_curve(
        wdbc_df,
        label_name=wdbc_csv_label_name,
        best_arch=[credit_input_dim, 2, 1],
        best_lambda=0.01,
        k=5,
        random_seed=42,
        positive_label=1,
        epochs=10000,
        learning_rate=0.05,
        train_size=None,
    )

    wdbc_best_result.to_csv(os.path.join(dir, "wdbc_learning_curce.csv"), index=False)

    make_graph_plt(
        wdbc_best_result,
        x_lab="train_size",
        y_lab="test_loss",
        title="WDBC learning curve",
        out_dir="../picture/",
        filename="wdbc_learning_curce.png",
    ) """
    ###########################################

    ############_____Loan_____#################
    """
    loan_df = read_csv(loan_csv_path)
    loan_df = prepossing_loan(loan_df)
    # loan_df.to_csv(os.path.join(dir, "loan_intermeditae_result.csv"), index=False)
    loan_dir = "../result/"
    loan_input_dim = loan_df.shape[1] - 1
    """

    """
    loan_result = train_test_on_fold(
        loan_df,
        label_name=wdbc_csv_label_name,
        arch_linux=[
            [loan_input_dim, 2, 1],
            [loan_input_dim, 4, 1],
            [loan_input_dim, 8, 1],
            [loan_input_dim, 16, 1],
            [loan_input_dim, 32, 16, 1],
            [loan_input_dim, 8, 4, 1],
            [loan_input_dim, 16, 8, 4, 1],
            [loan_input_dim, 16, 8, 4, 2, 1],
            [loan_input_dim, 32, 64, 32, 1],
        ],
        lambdas=[0.01, 0.05, 0.1, 0.3],
        k=5,
        random_seed=42,
        positive_label=1,
        epochs=50000,
        learning_rate=0.05,
    )

    loan_result.to_csv(os.path.join(dir, "loan_result.csv"), index=False)
    print(loan_result)
    """

    """
    loan_best_result = learning_curve(
        loan_df,
        label_name=wdbc_csv_label_name,
        best_arch=[loan_input_dim, 2, 1],
        best_lambda=0.3,
        k=5,
        random_seed=42,
        positive_label=1,
        epochs=20000,
        learning_rate=0.05,
        train_size=None,
    )

    loan_best_result.to_csv(os.path.join(dir, "loan_learning_curce.csv"), index=False)

    make_graph_plt(
        loan_best_result,
        x_lab="train_size",
        y_lab="test_loss",
        title="Loan learning curve",
        out_dir="../picture/",
        filename="loan_learning_curce.png",
    )
    """
    ################################################

    ##############____Raisins_____##################
    """
    raisian_csv_path = "../datasets/raisin.csv"
    raisian_csv_label_name = "label"

    raisian_df = read_csv(raisian_csv_path)
    dir = "../result/"
    os.makedirs(dir, exist_ok=True)
    raisian_input_dim = raisian_df.shape[1] - 1
    """

    """"
    raisian_result = train_test_on_fold(
        raisian_df,
        label_name=raisian_csv_label_name,
        arch_linux=[
            [raisian_input_dim, 2, 1],
            [raisian_input_dim, 4, 1],
            [raisian_input_dim, 8, 1],
            [raisian_input_dim, 16, 1],
            [raisian_input_dim, 8, 4, 1],
            [raisian_input_dim, 16, 8, 4, 1],
            [raisian_input_dim, 16, 8, 4, 2, 1],
        ],
        lambdas=[0.01, 0.05, 0.1, 0.3],
        k=5,
        random_seed=42,
        positive_label=1,
        epochs=30000,
        learning_rate=0.05,
    )

    raisian_result.to_csv(os.path.join(dir, "raisian_result.csv"), index=False)

    print(raisian_result)

    """

    """
    raisian_best_result = learning_curve(
        raisian_df,
        label_name=raisian_csv_label_name,
        best_arch=[raisian_input_dim, 16, 8, 4, 1],
        best_lambda=0.01,
        k=5,
        random_seed=42,
        positive_label=1,
        epochs=30000,
        learning_rate=0.05,
        train_size=None,
    )

    raisian_best_result.to_csv(
        os.path.join(dir, "raisian_learning_curce.csv"), index=False
    )

    make_graph_plt(
        raisian_best_result,
        x_lab="train_size",
        y_lab="test_loss",
        title="Raisian learning curve",
        out_dir="../picture/",
        filename="Raisian_learning_curce.png",
    )
    """

    ####################################################

    #################___titan___########################
    """
    titan_csv_path = "../datasets/titanic.csv"
    titan_csv_label_name = "label"

    titan_df = read_csv(titan_csv_path, label_name=titan_csv_label_name)
    titan_df = prepossing_loan(titan_df, label_name=titan_csv_label_name)
    # titan_df.to_csv(os.path.join(dir, "titan_intermeditae_result.csv"), index=False)

    dir = "../result/"
    os.makedirs(dir, exist_ok=True)
    titan_input_dim = titan_df.shape[1] - 1
    """

    """
    titan_result = train_test_on_fold(
        titan_df,
        label_name=titan_csv_label_name,
        arch_linux=[
            [titan_input_dim, 2, 1],
            [titan_input_dim, 4, 1],
            [titan_input_dim, 8, 1],
            [titan_input_dim, 16, 1],
            [titan_input_dim, 8, 4, 1],
            [titan_input_dim, 16, 8, 4, 1],
            [titan_input_dim, 16, 8, 4, 2, 1],
        ],
        lambdas=[0.01, 0.05, 0.1, 0.3],
        k=5,
        random_seed=42,
        positive_label=1,
        epochs=30000,
        learning_rate=0.05,
    )

    titan_result.to_csv(os.path.join(dir, "titan_result.csv"), index=False)

    print(titan_result)
    """
    """
    titan_best_result = learning_curve(
        titan_df,
        label_name=titan_csv_label_name,
        best_arch=[titan_input_dim, 16, 1],
        best_lambda=0.1,
        k=5,
        random_seed=42,
        positive_label=1,
        epochs=30000,
        learning_rate=0.05,
        train_size=None,
    )

    titan_best_result.to_csv(os.path.join(dir, "titan_learning_curce.csv"), index=False)

    make_graph_plt(
        titan_best_result,
        x_lab="train_size",
        y_lab="test_loss",
        title="Titan learning curve",
        out_dir="../picture/",
        filename="Titan_learning_curce.png",
    )
    """

    credit_csv_path = "../datasets/credit_approval.csv"
    credit_csv_label_name = "label"

    # loan_csv_path = "../datasets/loan.csv"
    # loan_csv_label_name = "label"

    #################_WDBC_#################
    credit_df = read_csv(credit_csv_path)
    credit_df = prepossing(credit_df)
    dir = "../result/"
    os.makedirs(dir, exist_ok=True)
    credit_input_dim = credit_df.shape[1] - 1

    credit_result = train_test_on_fold(
        credit_df,
        label_name=credit_csv_label_name,
        arch_linux=[
            [credit_input_dim, 2, 1],
            [credit_input_dim, 4, 1],
            [credit_input_dim, 8, 1],
            [credit_input_dim, 16, 1],
            [credit_input_dim, 8, 4, 1],
            [credit_input_dim, 16, 8, 4, 1],
            [credit_input_dim, 16, 8, 4, 2, 1],
        ],
        lambdas=[0.01, 0.05, 0.1, 0.3],
        k=10,
        random_seed=42,
        positive_label=1,
        epochs=20000,
        learning_rate=0.05,
    )

    credit_result.to_csv(os.path.join(dir, "credit_result.csv"), index=False)
    credit_result["index"] = range(1, len(credit_result) + 1)

    make_graph_plt(
        credit_result,
        x_lab="index",
        y_lab="acc",
        title="NN - Acc",
        out_dir="../picture/",
        filename="credit_nn_acc.png",
    )

    make_graph_plt(
        credit_result,
        x_lab="index",
        y_lab="f1",
        title="NN - F1 ",
        out_dir="../picture/",
        filename="credit_nn_f1.png",
    )

    print(credit_result)

    """
    wdbc_best_result = learning_curve(
        wdbc_df,
        label_name=wdbc_csv_label_name,
        best_arch=[credit_input_dim, 2, 1],
        best_lambda=0.01,
        k=5,
        random_seed=42,
        positive_label=1,
        epochs=10000,
        learning_rate=0.05,
        train_size=None,
    )

    wdbc_best_result.to_csv(os.path.join(dir, "wdbc_learning_curce.csv"), index=False)

    make_graph_plt(
        wdbc_best_result,
        x_lab="train_size",
        y_lab="test_loss",
        title="WDBC learning curve",
        out_dir="../picture/",
        filename="wdbc_learning_curce.png",
    ) """
    ###########################################

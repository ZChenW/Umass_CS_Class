import numpy as np


def print_row(arr):
    arr = np.array(arr).reshape(-1)
    return "[" + "   ".join(f"{x:.5f}" for x in arr) + "]"


def print_matrix(matrix):
    for row in matrix:
        print("  ".join(f"{x:.5f}" for x in row))


def print_initial(net, layer_idx):
    W = net.weights[layer_idx]
    b = net.bias[layer_idx]
    # Bias first, weight second
    bias_with_weight = np.concatenate([b.T, W.T], axis=1)
    return bias_with_weight


# 这个函数可以用于把activations转成带bias的格式
# 最后一行为原始输出层，不需要bias
def compute_forward(net, x):
    _, zs, acts = net.forward(x)

    a_with_bias_list = []

    for i in range(0, len(acts) - 1):
        a = acts[i]
        a_bias = np.concatenate([np.ones((a.shape[0], 1)), a], axis=1)
        a_with_bias_list.append(a_bias)

    a_with_bias_list.append(acts[-1])

    return a_with_bias_list, zs, acts


def compute_instance(net, x, y):
    y_pred, _, activations = net.forward(x)

    dZ = y_pred - y
    deltas = [None] * len(net.weights)
    deltas[-1] = dZ.copy()

    dWs = [None] * len(net.weights)
    dbs = [None] * len(net.weights)

    for l in reversed(range(len(net.weights))):
        A_prev = activations[l]

        dW = np.dot(A_prev.T, dZ)  # 单样本，所以不除 m
        db = dZ.copy()

        dWs[l] = dW
        dbs[l] = db

        if l > 0:
            dA_prev = np.dot(dZ, net.weights[l].T)
            dZ = dA_prev * (activations[l] * (1 - activations[l]))
            # delta是Error term
            deltas[l - 1] = dZ.copy()

    # 转成参考文件里的 Theta 梯度格式: [db | dW^T]
    grad_thetas = []
    for i in range(len(net.weights)):
        grad_theta = np.concatenate([dbs[i].T, dWs[i].T], axis=1)
        grad_thetas.append(grad_theta)

    return deltas, grad_thetas


# 对整个训练集求平均梯度，并加上正则化。
def compute_batch(net, X, Y, lambda_):
    m = X.shape[0]
    dWs = [np.zeros_like(w) for w in net.weights]
    dbs = [np.zeros_like(b) for b in net.bias]

    for i in range(m):
        x = X[i : i + 1]
        y = Y[i : i + 1]
        y_pred, _, activations = net.forward(x)

        dZ = y_pred - y
        for l in reversed(range(len(net.weights))):
            A_prev = activations[l]
            dW = np.dot(A_prev.T, dZ)
            db = dZ.copy()

            dWs[l] += dW
            dbs[l] += db

            if l > 0:
                dA_prev = np.dot(dZ, net.weights[l].T)
                dZ = dA_prev * (activations[l] * (1 - activations[l]))

    batch_grad_thetas = []
    for l in range(len(net.weights)):
        avg_dW = dWs[l] / m
        avg_db = dbs[l] / m

        # 正则化只加到非 bias weight
        avg_dW_reg = avg_dW + (lambda_ / m) * net.weights[l]

        grad_theta = np.concatenate([avg_db.T, avg_dW_reg.T], axis=1)
        batch_grad_thetas.append(grad_theta)

    return batch_grad_thetas

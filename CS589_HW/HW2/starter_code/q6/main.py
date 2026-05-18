from utils import load_training_set, load_test_set
from .nb import MultinomialNaiveBayes
import matplotlib.pyplot as plt
import os


def make_graph_plt(alphas, title, dir, filename=None, accs=None):
    plt.figure()
    plt.plot(alphas, accs, marker="o")
    plt.xlabel("alpha")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.xscale("log")
    plt.grid()
    #    plt.ylim(0, 1)
    # plt.xlim(0.90, 1.05)
    plt.savefig(os.path.join(dir, filename), dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前文件位置

    out_dir = os.path.join(BASE_DIR, "picture")
    os.makedirs(out_dir, exist_ok=True)

    percentage_positive_instances_train = 0.1
    percentage_negative_instances_train = 0.5

    percentage_positive_instances_test = 1.0
    percentage_negative_instances_test = 1.0

    (pos_train, neg_train, vocab) = load_training_set(
        percentage_positive_instances_train, percentage_negative_instances_train
    )
    (pos_test, neg_test) = load_test_set(
        percentage_positive_instances_test, percentage_negative_instances_test
    )

    nb = MultinomialNaiveBayes()
    nb.fit(pos_train, neg_train, vocab, alpha=1.0)
    metrix = nb.evaluate(pos_test, neg_test)
    print("Alpha 1: ", metrix)
    # print("Number of positive training instances:", len(pos_train))
    # print("Number of negative training instances:", len(neg_train))
    # print("Number of positive test instances:", len(pos_test))
    # print("Number of negative test instances:", len(neg_test))

# [nltk_data] Downloading package stopwords to /home/chakew/nltk_data...
# [nltk_data]   Package stopwords is already up-to-date!
# Alpha 1:  {'TP': 3707, 'FP': 158, 'FN': 11293, 'TN': 14842, 'accuracy': 0.6183, 'precision': 0.9591203104786546, 'recall': 0.24713333333333334}

from utils import load_training_set, load_test_set
from .nb import MultinomialNaiveBayes


if __name__ == "__main__":
    percentage_positive_instances_train = 0.2
    percentage_negative_instances_train = 0.2

    percentage_positive_instances_test = 0.2
    percentage_negative_instances_test = 0.2

    (pos_train, neg_train, vocab) = load_training_set(
        percentage_positive_instances_train, percentage_negative_instances_train
    )
    (pos_test, neg_test) = load_test_set(
        percentage_positive_instances_test, percentage_negative_instances_test
    )

    nb = MultinomialNaiveBayes()
    nb.fit(pos_train, neg_train, vocab)

    metrix = nb.evaluate(pos_test, neg_test)
    print(metrix)
    # print("Number of positive training instances:", len(pos_train))
    # print("Number of negative training instances:", len(neg_train))
    # print("Number of positive test instances:", len(pos_test))
    # print("Number of negative test instances:", len(neg_test))

# [nltk_data] Downloading package stopwords to /home/chakew/nltk_data...
# [nltk_data]   Package stopwords is already up-to-date!
# {'TP': 1529, 'FP': 402, 'FN': 1470, 'TN': 2659, 'accuracy': 0.691089108910891, 'precision': 0.7918177110305541, 'recall': 0.5098366122040681}

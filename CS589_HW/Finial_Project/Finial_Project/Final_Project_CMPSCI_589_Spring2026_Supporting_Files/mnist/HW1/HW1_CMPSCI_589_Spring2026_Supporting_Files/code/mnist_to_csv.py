from sklearn import datasets
import pandas as pd

digits = datasets.load_digits(return_X_y=True)

digits_dataset_X = digits[0]
digits_dataset_y = digits[1]

df = pd.DataFrame(digits_dataset_X)

df["label"] = digits_dataset_y

df.to_csv("../datasets/digits_dataset.csv", index=False)

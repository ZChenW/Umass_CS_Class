import pandas as pd

input_file = "../datasets/parkinsons.csv"
output_file = "parkinsons_label.csv"

df = pd.read_csv(input_file)
df = df.rename(columns={"Diagnosis": "label"})

df.to_csv(output_file, index=False)

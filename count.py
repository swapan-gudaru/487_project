import pandas as pd

labels = pd.read_csv('combined_sarcasm_dataset.csv')["Label"]
# print(labels)
sarcastic = 0
non_sarcastic = 0
for label in labels.values:
    if label == "1":
        sarcastic += 1
    elif label == "0":
        non_sarcastic += 1
print(f"Number of sarcastic comments: {sarcastic}")
print(f"Number of non-sarcastic comments: {non_sarcastic}")
print(f"Total comments: {sarcastic + non_sarcastic}")

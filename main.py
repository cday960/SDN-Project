import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from feature_selection import (
    get_top_features_rfe,
    get_top_features_corr,
    get_top_features_rf,
    get_top_features_mi,
    label_feature_correlation_heatmap,
)

"""
- Each label has 200 entries.
- Use data.loc[data[<column header>] == <header value>]
- Can extract specific features from above code using string indexing
"""


def extract_feature(dataframe, label, feature):
    result = []
    for _, row in dataframe.iterrows():
        if row["Label"] == label:
            result.append(row[feature])
    return result


def average_feature(dataframe, label, feature):
    total = 0
    count = 0
    for _, row in dataframe.iterrows():
        if row["Label"] == label:
            total += row[feature]
            count += 1
    return total / count


def my_print(data):
    for key, value in data.items():
        print(f"{'-'*50}\n{key}:")
        for x, y in value.items():
            print(f"\t{x}: {y}")


def histogram(data):
    histo = {}
    for value in data.values():
        for label in value.keys():
            if label not in histo:
                histo[label] = 0
            histo[label] += 1
    return histo


# Load data
# data = pd.read_csv("datasets/CICFlowMeter_Testing_Balanced.csv")
data = pd.read_csv("datasets/Custom_DNP3_Parser_Testing_Balanced.csv")
print(data.columns)

# Features want to drop
drop_features = [
    "Unnamed: 0.1",
    "Unnamed: 0",
    "Src Port",
]
data = data.drop(columns=drop_features, errors="ignore")

# Encode labels to numerical value to make processing easier
le = LabelEncoder()
data["Encoded_Label"] = le.fit_transform(data["Label"])

numerical_data = data.select_dtypes(include=[np.number])

attack_labels = data["Label"].unique()
attack_labels = [x for x in attack_labels if x != "NORMAL"]


top_features_corr = {
    label: get_top_features_corr(label, data, top_n=20)[0] for label in attack_labels
}

my_print(top_features_corr)

top_features_mi = {
    label: get_top_features_mi(label, data, top_n=20)[0] for label in attack_labels
}
my_print(top_features_mi)

# top_features_rf = {
#     label: get_top_features_rf(label, data, top_n=20)[0] for label in attack_labels
# }
# my_print(top_features_rf)


# rfe_results_by_attack = {
#     label: get_top_features_rfe(label, data, n_features_to_select=10)
#     for label in attack_labels
# }
# print(rfe_results_by_attack)

# Creates and saves a correlation heatmap between labels and features
# label_feature_correlation_heatmap(data)

# Creates a csv for the top features correlation
result = []

for label in attack_labels:
    top_features, _ = get_top_features_corr(label, data, top_n=20)

    for feature, corr in top_features.items():
        result.append({"Label": label, "Feature": feature, "Correlation": corr})

results_df = pd.DataFrame(result)
results_df.to_csv("./exported_data/top_correlated_features.csv")


# Creates a histogram
histo = histogram(top_features_corr)

# print("\n\nFrequency of features:")
# for key, value in histo.items():
#     print(f"\t{key}: {value}")

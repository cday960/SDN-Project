import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import (
    Dense,
    Input,
    Concatenate,
    BatchNormalization,
    Dropout,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from feature_selection import (
    get_top_features_rfe,
    get_top_features_corr,
    get_top_features_rf,
    get_top_features_mi,
    label_feature_correlation_heatmap,
    preprocess_data,
    random_split_features,
)

"""
- Each label has 200 entries.
- Use data.loc[data[<column header>] == <header value>]
- Can extract specific features from above code using string indexing
"""


def create_branch(input_dim):
    # model = models.Sequential(  # ~70% acc on 1000 epochs
    #     [
    #         Dense(64, activation="relu", input_dim=input_dim),
    #         Dense(32, activation="relu"),
    #         Dense(1, activation="sigmoid"),
    #     ]
    # )

    model = models.Sequential(  # ~85% acc on 100 epochs
        [
            Dense(128, activation="relu", input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
        ]
    )
    return model


def random_factoring_tree(x_train, splits, num_classes):
    inputs = []
    outputs = []

    for split in splits:
        branch_input = Input(shape=(len(split),))
        branch_model = create_branch(len(split))
        branch_output = branch_model(branch_input)
        inputs.append(branch_input)
        outputs.append(branch_output)

    # Combine outputs of ALL branches
    combined = Concatenate()(outputs)
    # Final classification
    # final_output = Dense(1, activation="sigmoid")(combined)
    final_output = Dense(num_classes, activation="sigmoid")(combined)

    model = models.Model(inputs=inputs, outputs=final_output)
    return model


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


def main():
    train_df = pd.read_csv("datasets/Custom_DNP3_Parser_Training_Balanced.csv")
    test_df = pd.read_csv("datasets/Custom_DNP3_Parser_Testing_Balanced.csv")

    drop_columns = [
        "Unnamed: 0.1",
        "Unnamed: 0",
        "source port",
        "source IP",
        "destination IP",
    ]
    target_column = "Label"

    x_train, y_train, label_encodings = preprocess_data(
        train_df, target_column, drop_columns
    )
    x_test, y_test, _ = preprocess_data(test_df, target_column, drop_columns)

    splits = random_split_features(x_train, num_splits=3)

    num_classes = y_train.shape[1]

    model = random_factoring_tree(x_train, splits, num_classes)

    # custom optimizer
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    # stock optimizer
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    x_train_branches = [x_train[:, split] for split in splits]
    x_test_branches = [x_test[:, split] for split in splits]

    model.fit(
        x_train_branches,
        y_train,
        epochs=500,
        batch_size=32,
        validation_data=(x_test_branches, y_test),
    )

    # loss, accuracy = model.evaluate(x_test_branches, y_test)
    loss, accuracy = model.evaluate(x_test_branches, y_test)
    print(f"Test loss: {loss}, Test accuracy: {accuracy}")

    y_pred = model.predict(x_test_branches)
    y_pred_binary = (y_pred > 0.5).astype(int)
    print(classification_report(y_test, y_pred_binary))

    for i, label in enumerate(label_encodings):
        print(f"{i}: {label}")
    # print(label_encodings)

    model.save("./models/new_model.h5")


if __name__ == "__main__":
    main()


# Load data
# data = pd.read_csv("datasets/CICFlowMeter_Testing_Balanced.csv")
# data = pd.read_csv("datasets/Custom_DNP3_Parser_Testing_Balanced.csv")
# print(data.columns)

# # Features want to drop
# drop_features = [
#     "Unnamed: 0.1",
#     "Unnamed: 0",
#     "Src Port",
# ]
# data = data.drop(columns=drop_features, errors="ignore")

# # Encode labels to numerical value to make processing easier
# le = LabelEncoder()
# data["Encoded_Label"] = le.fit_transform(data["Label"])

# numerical_data = data.select_dtypes(include=[np.number])

# attack_labels = data["Label"].unique()
# attack_labels = [x for x in attack_labels if x != "NORMAL"]


# top_features_corr = {
#     label: get_top_features_corr(label, data, top_n=20)[0] for label in attack_labels
# }

# my_print(top_features_corr)

# top_features_mi = {
#     label: get_top_features_mi(label, data, top_n=20)[0] for label in attack_labels
# }
# my_print(top_features_mi)

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
# result = []

# for label in attack_labels:
#     top_features, _ = get_top_features_corr(label, data, top_n=20)

#     for feature, corr in top_features.items():
#         result.append({"Label": label, "Feature": feature, "Correlation": corr})

# results_df = pd.DataFrame(result)
# results_df.to_csv("./exported_data/top_correlated_features.csv")


# Creates a histogram
# histo = histogram(top_features_corr)

# print("\n\nFrequency of features:")
# for key, value in histo.items():
#     print(f"\t{key}: {value}")

import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import joblib
from tensorflow.keras import models
from tensorflow.keras.layers import (
    Dense,
    Input,
    Concatenate,
    BatchNormalization,
    Dropout,
)
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from feature_selection import (
    get_top_features_rfe,
    get_top_features_corr,
    get_top_features_rf,
    get_top_features_mi,
    label_feature_correlation_heatmap,
    preprocess_isolate,
    preprocess_data,
    preprocess_data_sklearn,
    preprocess_isolate_columns,
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


def load_tf_model(path):
    loaded_model = models.load_model(path)
    return loaded_model


def train_cold_warm(epochs, learning_rate):
    train_df = pd.read_csv("datasets/Custom_DNP3_Parser_Training_Balanced.csv")
    test_df = pd.read_csv("datasets/Custom_DNP3_Parser_Testing_Balanced.csv")

    drop_columns = [
        "Unnamed: 0.1",
        "Unnamed: 0",
    ]
    target_column = "Label"

    x_train, y_train = preprocess_isolate(train_df, target_column, drop_columns)
    x_test, y_test = preprocess_isolate(test_df, target_column, drop_columns)

    model = tf.keras.Sequential(
        [
            Dense(64, activation="relu", input_dim=x_train.shape[1]),
            Dense(32, activation="relu"),
            Dense(2, activation="sigmoid"),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        x_train, y_train, epochs=epochs, batch_size=32, validation_data=(x_test, y_test)
    )

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Loss: {loss}, Accuracy: {accuracy}")


def train_random_factoring_tree_model(epochs, learning_rate, save_model=False):
    train_df = pd.read_csv("datasets/Custom_DNP3_Parser_Training_Balanced.csv")
    test_df = pd.read_csv("datasets/Custom_DNP3_Parser_Testing_Balanced.csv")

    drop_columns = [
        "Unnamed: 0.1",
        "Unnamed: 0",
    ]
    target_column = "Label"

    # x_train, y_train, mlb = preprocess_data(
    #     train_df, target_column, drop_columns, save_file=save_model
    # )
    # x_test, y_test, mlb = preprocess_data(test_df, target_column, drop_columns)

    x_train, y_train = preprocess_isolate_columns(train_df, target_column, drop_columns)
    x_test, y_test = preprocess_isolate_columns(test_df, target_column, drop_columns)

    splits = random_split_features(x_train, num_splits=3)

    # num_classes = y_train.shape[1]
    num_classes = 2

    model = random_factoring_tree(x_train, splits, num_classes)

    # custom optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    # stock optimizer
    # model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    x_train_branches = [x_train[:, split] for split in splits]
    x_test_branches = [x_test[:, split] for split in splits]

    model.fit(
        x_train_branches,
        y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(x_test_branches, y_test),
    )

    # loss, accuracy = model.evaluate(x_test_branches, y_test)
    loss, accuracy = model.evaluate(x_test_branches, y_test)
    print(f"Test loss: {loss}, Test accuracy: {accuracy}")

    y_pred = model.predict(x_test_branches)
    y_pred_binary = (y_pred > 0.5).astype(int)
    # class_report = classification_report(
    #     y_test, y_pred_binary, target_names=mlb.classes_
    # )

    class_report = classification_report(y_test, y_pred_binary)

    if save_model:
        model.save("./models/new_model.h5")

    return class_report


def random_forest_pipeline(n_estimators, save_model=False):
    dataset = pd.read_csv("datasets/CICFlowMeter_Testing_Balanced.csv")

    drop_columns = [
        "Unnamed: 0.1",
        "Unnamed: 0",
        # "source port",
        # "source IP",
        # "destination IP",
    ]
    target_column = "Label"

    (x_train, x_test, y_train, y_test), mlb = preprocess_data_sklearn(
        dataset, target_column, drop_columns
    )

    rf_model = OneVsRestClassifier(
        RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    )
    rf_model.fit(x_train, y_train)

    y_pred = rf_model.predict(x_test)

    print("Random forst classification report:")
    class_report = classification_report(y_test, y_pred, target_names=mlb.classes_)
    # print(classification_report(y_test, y_pred, target_names=mlb.classes_))

    if save_model:
        joblib.dump(rf_model, "./models/new_sklearn_model.pkl")

    return class_report


def main():
    # rft_report = train_random_factoring_tree_model(
    #     epochs=500, learning_rate=0.001, save_model=False
    # )
    # forest_report = random_forest_pipeline(n_estimators=100, save_model=True)

    # print("Random Factoring Tree:")
    # print(rft_report)
    # print("Random Forest Algorithm:")
    # print(forest_report)

    train_cold_warm(100, 0.00001)

    # model = models.load_model("./models/new_model.h5")
    # scaler = joblib.load("./models/standard_scaler.pkl")
    # mlb = joblib.load("./models/multi_label_binarizer.pkl")

    # test_df = pd.read_csv("datasets/Custom_DNP3_Parser_Testing_Balanced.csv")
    # target_column = "Label"
    # drop_columns = [
    #     "Unnamed: 0.1",
    #     "Unnamed: 0",
    # ]

    # x_test = test_df.drop(
    #     columns=drop_columns + [target_column], errors="ignore"
    # ).select_dtypes(include=["int64", "float64"])
    # y_test = test_df[target_column].str.split(",")

    # x_test, y_test, _ = preprocess_data(test_df, target_column, drop_columns)

    # splits = np.array_split(np.arange(x_test.shape[1]), 3)
    # x_test_scaled = scaler.transform(x_test)
    # x_test_branches = [x_test[:, split] for split in splits]

    # y_test_binarized = mlb.transform(y_test)

    # y_pred_probs = model.predict(x_test_branches)
    # y_pred = (y_pred_probs > 0.5).astype(int)
    # print(classification_report(y_test_binarized, y_pred, target_names=mlb.classes_))


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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def preprocess_data(df, target_column, drop_columns=None):
    """
    Drops unnecessary columns.
    Encodes the target column.
    Scales features.
    """

    df.columns = df.columns.str.strip()

    if drop_columns:
        df = df.drop(columns=drop_columns)

    # print(df.columns)

    # Encode target var for classification
    # le = LabelEncoder()
    # df[target_column] = le.fit_transform(df[target_column])

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df[target_column].str.split(","))

    label_encodings = []
    for _, label in enumerate(mlb.classes_):
        label_encodings.append(label)

    # Creates a list of all columns that are numerical so we can
    # encode all of the features
    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns
    # numerical_features = numerical_features.drop(target_column)

    # Separate features and target
    x = df[numerical_features]
    # y = df[target_column]

    # Scale features
    """
    This standard scaler standardizes the features to have a mean of 0 and a std of 1.
    Helps deal with flow duration dominating the model.
    """
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    return x_scaled, y, label_encodings


def random_split_features(x, num_splits=3):
    """
    Randomly split up features into (by default) 3 groups
    for tree branches.
    """
    n_features = x.shape[1]
    feature_indices = np.arange(n_features)
    np.random.shuffle(feature_indices)
    splits = np.array_split(feature_indices, num_splits)
    return splits


def get_top_features_corr(attack_label, data, top_n=10):
    # Select attack and normal flows
    attack_data = data[data["Label"] == attack_label].select_dtypes(include=[np.number])
    normal_data = data[data["Label"] == "NORMAL"].select_dtypes(include=[np.number])

    # Combine normal and attack flows
    combined = pd.concat([normal_data, attack_data])

    # Create target column: 0 for NORMAL, 1 for specific attacks
    combined["Target"] = [0] * len(normal_data) + [1] * len(attack_data)

    # Calculate Pearson correlation
    correlation_matrix = combined.corr()

    plt.figure(figsize=(30, 26))
    sns.heatmap(
        correlation_matrix,
        annot=False,
        cmap="coolwarm",
        cbar=True,
        linewidths=0.3,
        linecolor="gray",
        xticklabels=True,
        yticklabels=True,
    )
    plt.title("Feature to Feature interaction")
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    # plt.show()
    plt.savefig("./figures/feature-to-feature.png", format="png", dpi=300)

    # Get correlations with the target and sort by abs value
    target_correlation = (
        correlation_matrix["Target"].abs().sort_values(ascending=False).drop("Target")
    )

    # CSV stuff
    # csv_data = target_correlation.head(top_n).reset_index()
    # csv_data.columns = ["Feature", "Correlation"]
    # csv_data.to_csv("./data/top_correlated_features.csv", index=False)

    # Drop "Target" itself and return top N features
    top_features = target_correlation.head(top_n)
    bottom_features = target_correlation.tail(top_n)

    # return target_correlation.drop("Target").head(top_n)
    return [top_features.to_dict(), bottom_features.to_dict()]


def get_top_features_mi(attack_label, data, top_n=10):
    # Select attack and normal flows
    attack_data = data[data["Label"] == attack_label].select_dtypes(include=[np.number])
    normal_data = data[data["Label"] == "NORMAL"].select_dtypes(include=[np.number])

    # Combine normal and attack flows
    combined = pd.concat([normal_data, attack_data])
    target = [0] * len(normal_data) + [1] * len(attack_data)

    # Compute mutual information scores
    mi_scores = mutual_info_classif(combined, target, discrete_features="auto")

    # convert scores into dataframe
    mi_scores_df = pd.Series(mi_scores, index=combined.columns).sort_values(
        ascending=False
    )

    top = mi_scores_df.head(top_n)
    bot = mi_scores_df.tail(top_n)
    # print(type(top))

    return [top, bot]


def get_top_features_rf(attack_label, data, top_n=10):
    # Select attack and normal flows
    attack_data = data[data["Label"] == attack_label].select_dtypes(include=[np.number])
    normal_data = data[data["Label"] == "NORMAL"].select_dtypes(include=[np.number])

    # Combine normal and attack flows
    combined = pd.concat([normal_data, attack_data])
    target = [0] * len(normal_data) + [1] * len(attack_data)

    # Fit random forest and get importances
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    rf.fit(combined, target)

    feature_importances = pd.Series(
        rf.feature_importances_, index=combined.columns
    ).sort_values(ascending=False)

    top = feature_importances.head(top_n)
    bot = feature_importances.tail(top_n)

    return [top, bot]


def get_top_features_rfe(attack_label, data, n_features_to_select=10):
    # Select attack and normal flows
    attack_data = data[data["Label"] == attack_label].select_dtypes(include=[np.number])
    normal_data = data[data["Label"] == "NORMAL"].select_dtypes(include=[np.number])

    # Combine normal and attack flows
    combined = pd.concat([normal_data, attack_data])
    target = [0] * len(normal_data) + [1] * len(attack_data)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Initialize rfe with num of feat to select
    rfe = RFE(estimator=rf, n_features_to_select=n_features_to_select, step=1)

    rfe.fit(combined, target)

    selected_features = combined.columns[rfe.support_]
    feature_ranking = pd.Series(rfe.ranking_, index=combined.columns).sort_values()

    return selected_features, feature_ranking.head(10)


def label_feature_correlation_heatmap(data):
    binary_labels = pd.get_dummies(data["Label"])

    numerical_data = data.select_dtypes(include=[np.number])

    correlation_matrix = pd.concat([numerical_data, binary_labels], axis=1).corr()

    label_correlation = correlation_matrix.loc[
        numerical_data.columns, binary_labels.columns
    ]

    plt.figure(figsize=(30, 26))
    sns.heatmap(
        label_correlation,
        annot=True,
        cmap="coolwarm",
        cbar=True,
        linewidths=0.3,
        linecolor="gray",
    )

    plt.title("Feature Label Correlation")
    plt.xticks(rotation=45, ha="right", fontsize=14)
    plt.yticks(fontsize=14)
    plt.subplots_adjust(left=0.12)
    plt.subplots_adjust(right=0.15)
    plt.tight_layout()
    plt.savefig("./figures/feature-to-label.png", format="png", dpi=300)

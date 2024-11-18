import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier


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

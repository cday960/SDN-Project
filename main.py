# import sklearn
import numpy as np
import pandas as pd

"""
- Each label has 199 entries.
"""

data = pd.read_csv("CICFlowMeter_Testing_Balanced.csv")
labels = {}

for label in data["Label"]:
    if label not in labels:
        labels[label] = {"Amount": 0}
    else:
        labels[label]["Amount"] += 1
        # print(labels[label]["Amount"])

# print(labels)
print(data.head)
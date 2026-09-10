"""
It builds the initial dataset, encodes it, and trains a
DecisionTreeClassifier. Also loads any previously-saved user data so
the model keeps improving across runs. Eventually it will be modeled specific to the user's preferences.
"""

import os
from sklearn import tree
import pandas as pd

# Column order used everywhere data is created
categories = ["outlook", "temperature", "humidity", "windy", "play"]

# The allowed value for each input feature and used to validate user input
# so typos/case differences don't  produce a garbage encoding.
ALLOWED_VALUES = {
    "outlook": {"Sunny", "Overcast", "Rain"},
    "temperature": {"Hot", "Mild", "Cool"},
    "humidity": {"High", "Normal"},
    "windy": {"True", "False"},
}

# Where we persist any data the user adds, so it survives between runs
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tennis_data.csv")

# Original 14 rows of training data | outlook, temperature, humidity, windy, play
Data = [["Sunny",    "Hot",  "High",   "False", "No"],
        ["Sunny",    "Hot",  "High",   "True",  "No"],
        ["Overcast", "Hot",  "High",   "False", "Yes"],
        ["Rain",     "Mild", "High",   "False", "Yes"],
        ["Rain",     "Cool", "Normal", "False", "Yes"],
        ["Rain",     "Cool", "Normal", "True",  "No"],
        ["Overcast", "Cool", "Normal", "True",  "Yes"],
        ["Sunny",    "Mild", "High",   "False", "No"],
        ["Sunny",    "Cool", "Normal", "False", "Yes"],
        ["Rain",     "Mild", "Normal", "False", "Yes"],
        ["Sunny",    "Mild", "Normal", "True",  "Yes"],
        ["Overcast", "Mild", "High",   "True",  "Yes"],
        ["Overcast", "Hot",  "Normal", "False", "Yes"],
        ["Rain",     "Mild", "High",   "True",  "No"]]


def load_initial_dataframe():
    """
    Load previously-saved user data if it exists, otherwise fall back to
    the original 14-row dataset. This is what makes the model's learning
    persist across program runs instead of resetting every time.
    """
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE, dtype=str)
    return pd.DataFrame(Data, columns=categories)


def save_dataframe(df):
    #Persist the current dataset to disk so it's available next run.
    df.to_csv(DATA_FILE, index=False)


def encode_features(df):
    #One-hot encode the 4 input features. sklearn needs numbers, not strings.
    return pd.get_dummies(df[["outlook", "temperature", "humidity", "windy"]])


def encode_labels(df):
    #Turn the play column into 1 (Yes) / 0 (No).
    return (df["play"] == "Yes").astype(int)


def retrain(df):
    """
    Single source of truth for 'turn a dataframe into a fitted model'.
    Both the prediction flow and the manual-input flow call this so the
    encoding/fitting logic never drifts out of sync between the two.
    Returns (X, y, clf) so callers can keep using the same variable names.
    """
    X = encode_features(df)
    y = encode_labels(df)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    return X, y, clf


# Build the initial trained model when this module is imported
df = load_initial_dataframe()
X, y, clf = retrain(df)
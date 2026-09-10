"""
UI side of the program that interacts with the user. It lets us view
predictions and add new data, and it slowly fits the model better to
the specific user's own observations over time.
"""

import pandas as pd
import tennistree
from tennistree import categories, ALLOWED_VALUES, retrain, save_dataframe


def prompt_for_value(field_name):
    """
    Ask for a single field's value, keep re-asking until the answer is
    one of the values the model actually knows about. Case and stray
    whitespace are normalized so "sunny " and "Sunny" both work. although would be a lot better
    if i simply created a button instead of asking for input, but i wanted to keep it simple and text-based.
    """
    allowed = ALLOWED_VALUES[field_name]
    while True:
        raw = input(f"{field_name.capitalize()} {sorted(allowed)}: ").strip()
        # Normalize casing: "sunny" -> "Sunny", "true"/"false" handled too
        normalized = raw.capitalize() if field_name != "windy" else raw.capitalize()
        if normalized in allowed:
            return normalized
        print(f"  Sorry, that's not a recognized value. Please choose from {sorted(allowed)}.")


def prompt_for_outcome():
    #Ask whether it was a good or bad day, re-asking until we get 0 or 1.
    while True:
        answer = input("Was that correct? Write 0 for bad day, 1 for good day: ").strip()
        if answer == "0":
            return "No"
        if answer == "1":
            return "Yes"
        print("  Please enter 0 or 1.")


def collect_new_row():
    #Prompt for all four features and return them as a plain dict.
    return {
        "outlook": prompt_for_value("outlook"),
        "temperature": prompt_for_value("temperature"),
        "humidity": prompt_for_value("humidity"),
        "windy": prompt_for_value("windy"),
    }


def main():
    df = tennistree.df
    X = tennistree.X
    y = tennistree.y
    clf = tennistree.clf

    print("Hello! This program predicts whether it's a good day for tennis, "
          "and learns from your feedback over time.")

    while True:
        choice = input("\nWould you like a prediction, to input data, or to quit? "
                        "Write 'P' for prediction, 'I' for input, 'Q' to quit: ").strip().upper()

        if choice == "P":
            print("Please tell me today's outlook, temperature, humidity, and windiness.")
            new_values = collect_new_row()
            new_row = pd.DataFrame([new_values], columns=["outlook", "temperature", "humidity", "windy"])

            # Encode the same way the training data was encoded, so the tree can read it
            new_encoded = pd.get_dummies(new_row).reindex(columns=X.columns, fill_value=0)

            prediction = clf.predict(new_encoded)
            print("Prediction:", "Good day for tennis!" if prediction[0] == 1 else "Not a good day for tennis.")

            # Let the user correct the model, then fold that row into the dataset
            new_row["play"] = prompt_for_outcome()
            df = pd.concat([df, new_row], ignore_index=True)
            X, y, clf = retrain(df)
            save_dataframe(df)

        elif choice == "I":
            print("Please enter today's data.")
            new_values = collect_new_row()
            new_values["play"] = prompt_for_outcome()
            new_row = pd.DataFrame([new_values], columns=categories)

            df = pd.concat([df, new_row], ignore_index=True)
            X, y, clf = retrain(df)
            save_dataframe(df)

        elif choice == "Q":
            print("Goodbye!")
            break

        else:
            print("Sorry, I didn't recognize that. Please write 'P', 'I', or 'Q'.")


if __name__ == "__main__":
    main()
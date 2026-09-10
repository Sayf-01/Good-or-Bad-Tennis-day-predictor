from tennistree import X, y, df, clf, Data, categories, retrain, ALLOWED_VALUES
import pandas as pd


def test_data_has_14_rows():
    # Arrange: nothing to set up, Data already exists
    # Act: nothing to run, we're just checking the data itself
    # Assert: check what we expect
    assert len(Data) == 14


def test_columns_match_categories():
    assert list(df.columns) == categories


def test_encoding_produces_only_numbers():
    # every value in X should be 0 or 1 (one-hot encoded), never a raw string
    assert X.isin([0, 1, True, False]).all().all()


def test_labels_are_binary():
    # y should only ever contain 0s and 1s
    assert set(y.unique()) <= {0, 1}


def test_model_predicts_known_row_correctly():
    # Arrange: take the very first row of training data - Sunny/Hot/High/False -> No
    first_row_encoded = X.iloc[[0]]
    # Act
    prediction = clf.predict(first_row_encoded)
    # Assert: the model should correctly remember data it was trained on
    assert prediction[0] == 0   # 0 means "No"


def test_model_only_outputs_0_or_1():
    predictions = clf.predict(X)
    assert set(predictions) <= {0, 1}


def test_retrain_is_consistent_with_module_level_fit():
    # retrain() should produce the same X/y shapes as the module-level training
    X2, y2, clf2 = retrain(df)
    assert X2.shape == X.shape
    assert list(y2) == list(y)


def test_unseen_category_gets_zeroed_not_crash():
    # A category value the model has never seen (e.g. a typo) should not
    # blow up - get_dummies + reindex should just zero out the unknown column.
    bad_row = pd.DataFrame([["Foggy", "Hot", "High", "False"]],
                           columns=["outlook", "temperature", "humidity", "windy"])
    encoded = pd.get_dummies(bad_row).reindex(columns=X.columns, fill_value=0)
    assert encoded.shape[1] == X.shape[1]
    # every known outlook column should read 0 since "Foggy" isn't one of them
    outlook_cols = [c for c in X.columns if c.startswith("outlook_")]
    assert encoded[outlook_cols].sum(axis=1).iloc[0] == 0


def test_allowed_values_cover_all_training_data():
    # sanity check: every value seen in the original training data must be
    # in ALLOWED_VALUES, or prompt_for_value would reject valid historical data
    for _, row in df.iterrows():
        for field in ["outlook", "temperature", "humidity", "windy"]:
            assert row[field] in ALLOWED_VALUES[field]
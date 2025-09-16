import numpy as np
import pandas as pd

'''
class LogisticRegression:
    def __init__(self):

    def prepare(self):

    def fit(self):

    def predict(self):

    def score(self):
'''

if __name__ == "__main__":
    columns = [
        'ID', 'Diagnosis',
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]

    df = pd.read_csv('wdbc.data', header=None, names=columns, dtype=str)

    df.drop(columns=['ID'], inplace=True) # drops id column

    missing_rows = df[df.isin(['?', 'NA', 'na', '']).any(axis=1)]  # checks null values
    print(f"Rows with null values: {len(missing_rows)}")

    df.replace(['?','NA', 'na', ''], pd.NA, inplace=True) # replace null values with NA identifier

    num_cols = df.columns.difference(['Diagnosis'])
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') # convert columns to numeric values

    df.dropna(inplace=True) # remove null values
    print(f"Rows remaining after drop of the null values: {len(df)}")
    for col in num_cols:
        df = df[df[col] >= 0]

    # check if there are still null values
    assert df.isna().sum().sum() == 0, "There are still some null values."

    df['Diagnosis'] = df['Diagnosis'].astype('category')
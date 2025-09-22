import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, add_bias = True)
    self.add_bias = add_bias
    pass

    def fit(self,x,y):
        if x.dim == 1:
            x = x[:,None]
        N = x.shape[0]
        if self.add_bias:
            x = np.column_stack ([x,np.ones(N)])
        self.w = np.linalg.lstsq(x,y)[0]
        return self

    def predict(self,x)
        if self.add_bias:
            x = np.column_stack ([x,np.ones(N)])
        yh = x@self.w
        return yh
    


class LinearRegression:
    '''
        Constructor for the linear regression with analytical solution. It uses bias. It also
        initializes the weight, mean and standard deviation.
    '''
    def __init__(self, add_bias):
        self.add_bias = add_bias  # bias to prepend a column of ones (the intercept term)
        self.w = None  # weight/coefficient
        self.mean = None  # used for standardisation
        self.std = None  # standard deviation


    def prepare(self, x: pd.DataFrame) -> pd.DataFrame:
        '''
            Preparation method to ensure X is a float DataFrame, add a bias if it is true and standardise the X.
        '''
        x = x.copy()
        x = x.astype('float64')

        if self.mean is None:  # standardisation
            self.mean = x.mean()
            self.std = x.std(ddof=0)
            self.std.replace(0, 1, inplace=True)  # guard against division by zero

        x = (x - self.mean) / self.std  # standardisation formula

        if self.add_bias:  # adding bias
            x['bias'] = 1.0

        return x


    def fit(self, x: pd.DataFrame, y: pd.Series) -> "LinearRegression":
        '''
            Fit method to fit X and Y datas through pandas and train the linear model by analytical solution.
            It uses pandas DataFrame for the X and Series for the Y. It uses the linear regression formula
            to calculate weight
        '''
        x = self.prepare(x)
        y = pd.Series(y).astype("float64")

        # convert to numpy for speed
        x_np = x.to_numpy()  # n_samples, n_features
        y_np = y.to_numpy()[:, None]  # n_samples, 1

        # w = (X^T*X)^-1*X^T*Y
        xt_x = x_np.T.dot(x_np)
        xt_y = x_np.T.dot(y_np)
        w_np = np.linalg.pinv(xt_x).dot(xt_y)  # n_features, 1

        # store weights back as a pandas series
        self.w = pd.Series(
            w_np.ravel(), # flattens the array into 1-D array
            index=x.columns
        )
        return self


    def predict(self, x: pd.DataFrame) -> pd.Series:
        '''
            Predict method is used to test trained data to do Y prediction by multiplying X and weight vectors.
        '''
        if self.w is None:  # if weight is empty, throw error
            raise RuntimeError("Model is not fitted yet. Call `fit` first.")

        x = self.prepare(x)  # standardisation and adding bias through prepare method
        return x.dot(self.w)

    def score(self, x: pd.DataFrame, y: pd.Series) -> float:
        '''
            This method is used to calculate coefficient of determination to assess the goodness
            of fit from the linear regression model
        '''
        y_pred = self.predict(x)  # predicts Y value with X predict method.
        y = pd.Series(y).astype('float64')
        ss_res = ((y - y_pred) ** 2).sum()
        # sum of squared residuals, residuals are difference between Y values and Y prediction values
        ss_tot = ((y - y.mean()) ** 2).sum()
        # total sum of squares, uses the difference between Y values and Y mean value
        return 1.0 - ss_res / ss_tot


if __name__ == "__main__":
    df = pd.read_csv('parkinsons_updrs.data', dtype=str)

    df.drop(columns=['subject#'], inplace=True)  # drops subject# column

    missing_rows = df[df.isin(['?', 'NA', 'na', '']).any(axis=1)] # checks null values
    print(f"Rows with null values: {len(missing_rows)}")

    df.replace(['?','NA', 'na', ''], pd.NA, inplace=True) # replace null values with NA identifier

    num_cols = [
        'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS',
        'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
        'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'
    ]

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') # convert columns to numeric values

    df.dropna(inplace=True) # remove null values
    print(f"Rows remaining after drop of the null values: {len(df)}\n")

    # sanity checks for data validity - realistic parkinson data range estimations
    df = df[(df['age'] >= 18) & (df['age'] <= 95)]
    df = df[(df['motor_UPDRS'] >= 0) & (df['motor_UPDRS'] <= 100)]
    df = df[(df['total_UPDRS'] >= 0) & (df['total_UPDRS'] <= 100)]
    df = df[(df['Jitter(%)'] >= 0) & (df['Jitter(%)'] <= 10)]
    df = df[(df['Shimmer(dB)'] >= 0) & (df['Shimmer(dB)'] <= 10)]

    print(f"Rows after sanity checks: {len(df)}")

    # check if there are still null values
    assert df.isna().sum().sum() == 0, "There are still some null values."

    # split the X and Y values
    target = 'total_UPDRS'
    x = df.drop(columns=[target])
    y = df[target]

    # train / test splitting (80 / 20)
    n_train = int(0.8 * len(x))
    x_train, x_test = x.iloc[:n_train], x.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

    # training of the model
    model = LinearRegression(add_bias=True)
    model.fit(x_train, y_train)

    # evaluation of the model
    print("\nR² on training data:", model.score(x_train, y_train))
    print("R² on testing data:", model.score(x_test, y_test))

    # predict Y values using the trained data
    preds = model.predict(x_test)
    print("\nFirst 10 predictions:")
    print(preds.head(10))

    # weight report
    print("\nWeights from the model:")
    print(model.w)
import numpy as np
import pandas as pd

class LinearRegression:
    '''
        Constructor for the linear regression with mini‑batch stochastic gradient descent. It uses learning rate,
        iteration number, batch size, bias and verbose. It also initializes the weight, mean and standard deviation.
    '''
    def __init__(self, lr, n_iter, batch_size, add_bias, verbose):
        self.lr = lr  # learning rate
        self.n_iter = n_iter  # number of gradient-descent iterations
        self.batch_size = batch_size  # row number for each gradient step
        self.add_bias = add_bias  # bias to prepend a column of ones (the intercept term)
        self.verbose = verbose  # if true, prints the mean‑squared error every 100 iterations
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
            Fit method to fit X and Y datas through pandas and train the linear model by gradient descent.
            It uses pandas DataFrame for the X and Series for the Y. For the n iterations, it returns batch X and Y values
            from random subset of indices calculates gradient from differences between predicted Y and batch Y values and
            calculates the weight. If verbose it prints the mean square error for each 100 iterations.
        '''
        x = self.prepare(x)  # standardisation and adding bias through prepare method
        y = pd.Series(y).astype('float64')  # check if Y is series.

        x_np = x.to_numpy()
        y_np = y.to_numpy()

        n_samples, n_features = x_np.shape # n samples
        w_np = np.zeros(n_features)   # initialize weight as zero
        batch_size = self.batch_size
        # defines n samples as batch size if size is none or bigger than n samples
        if batch_size is None or batch_size >= n_samples:
            batch_size = n_samples

        # number of batches per iteration
        n_batches = int(np.ceil(n_samples / batch_size))

        for epoch in range(1, self.n_iter + 1):
            shuffled_idx = np.random.permutation(n_samples) # random permutation of the indices
            for b in range(n_batches):
                start = b * batch_size
                end = start + batch_size
                idx = shuffled_idx[start:end]

                x_batch = x_np[idx]
                y_batch = y_np[idx]
                # it returns X and Y batch values from a randomly permuted indices from start to end

                y_pred = x_batch.dot(w_np)
                # makes Y prediction value for X batch value by multiplying X and weight vectors.

                error = y_batch - y_pred  # error is difference between Y batch value and Y prediction value
                grad = -2 * x_batch.T.dot(error) / batch_size
                # gradient is calculated by multiplication of error, transposed X batch value and -2 divided by batch size

                w_np -= self.lr * grad  # weight is decreased by multiplication of learning rate and gradient

            # if verbose, it calculates the mean squared error every 100 iterations and displays it
            if self.verbose and epoch % 100 == 0:
                y_full_pred = x.dot(w_np)
                mse = ((y_np - y_full_pred) ** 2).mean()
                print(f"Iter {epoch:5d} | MSE: {mse:.6f}")

        self.w = pd.Series(w_np, index=x.columns) # store weights back as a pandas series
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
    model = LinearRegression(lr=0.0001, n_iter=5000, batch_size=64, add_bias=True, verbose=True)
    # other values could be used, for example (lr=0.01, n_iter=2000, batch_size=None, add_bias=True, verbose=False)
    model.fit(x_train, y_train)

    # evaluation of the model
    print("\nR² on training data:", model.score(x_train, y_train))
    print("R² on testing data:", model.score(x_test, y_test))

    # predict Y values using the trained data
    preds = model.predict(x_test)
    print("\nFirst 10 predictions:")
    print(preds.head(10))
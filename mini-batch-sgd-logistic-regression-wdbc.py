import numpy as np
import pandas as pd


class LogisticRegression:
    '''
        Constructor for the logistic regression with gradient descent. It uses learning rate, iteration number,
        tolerance and verbose. It also initializes the weight, loss, x, y, mean and std.
    '''

    def __init__(self, learning_rate: float, n_iter: int, batch_size: int, tolerance: float, verbose: bool) -> None:
        self.lr = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.tol = tolerance
        self.verbose = verbose
        self.w: np.ndarray | None = None         # weight/coefficient (bias as first element)
        self.loss: list[float] = []              # loss per iteration
        self.x: np.ndarray | None = None         # matrix of inputs after standardisation
        self.y: np.ndarray | None = None         # target vector
        self.mean: np.ndarray | None = None      # used for standardisation
        self.std: np.ndarray | None = None       # standard deviation

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid method for the logistic regression method."""
        return 1.0 / (1.0 + np.exp(-z)) # 1/(1+exp(-z))

    @staticmethod
    def cost(y: np.ndarray, p: np.ndarray) -> float:
        """Cross‑entropy loss is used for the cost calculation"""
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def prepare(self, df: pd.DataFrame, target_col: str) -> None:
        """

        Preparation method splits df into x and y. It does define X and Y values from the dataframe and target column.
        Then it does standardisation, adds bias and initializes the weight/coefficient.

        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

        self.y = df[target_col].values.astype(np.int64)

        x_raw = df.drop(columns=[target_col]).values.astype(np.float64)

        # standardisation
        self.mean = x_raw.mean(axis=0)
        self.std = x_raw.std(axis=0)
        self.std[self.std == 0] = 1.0

        x_scaled = (x_raw - self.mean) / self.std  # standardisation formula


        bias = np.ones((x_scaled.shape[0], 1), dtype=np.float64)  # adding bias
        self.x = np.hstack((bias, x_scaled))

        self.w = np.zeros(self.x.shape[1], dtype=np.float64) # initialize weight as zero

    def fit(self) -> None:
        """

        Fit method to fit X and Y datas through pandas and train the linear model by gradient descent.
        For the n iterations, it finds probabilities through sigmoid of linear prediction and does the
        gradient to calculate the loss.

        """
        if self.x is None or self.y is None: # if x or y are empty, throw error
            raise RuntimeError("Model is not fitted yet. Call `prepare` first.")

        n_samples = self.x.shape[0]
        batch_size = self.batch_size or n_samples

        # number of batches per iteration
        n_batches = int(np.ceil(n_samples / batch_size))

        for epoch in range(1, self.n_iter + 1):
            shuffled_idx = np.random.permutation(n_samples) # random permutation of the indices
            for b in range(n_batches):
                start = b * batch_size
                end   = min(start + batch_size, n_samples)
                idx = shuffled_idx[start:end]

                x_batch = self.x[idx]
                y_batch = self.y[idx]


                z = x_batch.dot(self.w)
                p = self.sigmoid(z)

                grad = x_batch.T.dot(p - y_batch) / y_batch.size # gradient calculation formula
                self.w -= self.lr * grad # gradient multiplied by learning rate is removed from weight

            # cost is calculated through cross‑entropy and added for the current range
            loss = self.cost(self.y, self.sigmoid(self.x.dot(self.w)))
            self.loss.append(loss)

            # if verbose, it shows the loss every 100 iterations and displays it
            if self.verbose and epoch % 100 == 0:
                print(f"Iter {epoch:4d} – loss: {loss:.6f}")

            # tests whether the absolute change in loss is smaller than the tolerance
            if epoch > 1 and abs(self.loss[-2] - loss) < self.tol:
                if self.verbose:
                    print(f"Converged after {epoch} iterations.")
                break

    def predict(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
            Predict method is used to test trained data to do Y prediction by multiplying X and weight vectors
            and then calculates the model probability by applying sigmoid function.
        """
        if isinstance(x, pd.DataFrame): # verifies value type
            x = x.values.astype(np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        z = x.dot(self.w)
        probs = self.sigmoid(z) # probability calculation through sigmoid method
        return (probs >= 0.5).astype(int) # 0.5 is commonly used to define positivity of the probability

    def score(self, x: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> float:
        """
            This method is used to calculate mean accuracy with the prediction of Y and actual Y values.
        """
        y_pred = self.predict(x)
        y_true = np.asarray(y).astype(int)
        return np.mean(y_pred == y_true) # mean is calculated if Y values match

if __name__ == "__main__":
    columns = [
        'ID', 'Diagnosis',
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavitymean', 'concave_points_mean', 'symmetrymean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavityse', 'concave_points_se', 'symmetryse', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavityworst', 'concave_points_worst', 'symmetryworst', 'fractal_dimension_worst'
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
    print(f"Rows remaining after drop of the null values: {len(df)}\n")
    for col in num_cols:
        df = df[df[col] >= 0]

    # sanity checks for data validity - max tumor sizes possible
    df = df[(df['radius_mean'] > 0) & (df['radius_mean'] <= 30)]
    df = df[(df['radius_worst'] > 0) & (df['radius_worst'] <= 30)]
    df = df[(df['texture_mean'] >= 0) & (df['texture_mean'] <= 100)]
    df = df[(df['texture_worst'] >= 0) & (df['texture_worst'] <= 100)]
    df = df[(df['perimeter_mean'] > 0) & (df['perimeter_mean'] <= 200)]
    df = df[(df['perimeter_worst'] > 0) & (df['perimeter_worst'] <= 200)]
    df = df[(df['area_mean'] > 0) & (df['area_mean'] <= 600)]
    df = df[(df['area_worst'] > 0) & (df['area_worst'] <= 600)]

    # check if there are still null values
    assert df.isna().sum().sum() == 0, "There are still some null values."

    # making diagnosis numeric
    df["Diagnosis"] = df["Diagnosis"].map({"M": 1, "B": 0}).astype("category")

    rng = np.random.default_rng(seed=42)
    n_samples = len(df)
    indices = rng.permutation(n_samples)
    train_size = int(0.8 * n_samples)

    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # training of the model
    model = LogisticRegression(learning_rate=0.00005, n_iter=5000, batch_size=64, tolerance=1e-6, verbose=True)
    # other values could be used, for example (lr=0.01, n_iter=2000, tolerance=1e-3, verbose=False)
    model.prepare(df_train, target_col="Diagnosis")
    model.fit()

    # evaluation of the model
    train_acc = model.score(model.x, model.y)
    print(f"\nMean accuracy on training data: {train_acc:.4f}")

    # copied prepare method for building test X data
    x_test_raw = df_test.drop(columns=['Diagnosis']).values.astype(np.float64)
    x_test_scaled = (x_test_raw - model.mean) / model.std
    bias_test = np.ones((x_test_scaled.shape[0], 1), dtype=np.float64)
    X_test = np.hstack((bias_test, x_test_scaled))
    y_test = df_test['Diagnosis'].values.astype(int)
    test_acc = model.score(X_test, y_test)
    print(f"Mean accuracy on testing data: {test_acc:.4f}")

    # predict Y values using the trained data
    first_10 = X_test[:10]
    y_hat = model.predict(first_10)
    print("\nFirst 10 predictions:", y_hat.ravel())

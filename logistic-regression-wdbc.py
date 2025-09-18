import numpy as np
import pandas as pd

class LogisticRegressionGD:
    """Binary logistic regression trained with batch gradient descent."""
    def __init__(self,
                 learning_rate: float = 0.01,
                 n_iter: int = 1000,
                 tolerance: float = 1e-5,
                 verbose: bool = False):
        """
        Parameters
        ----------
        learning_rate : float
            Step size for weight updates.
        n_iter : int
            Maximum number of iterations.
        tolerance : float
            Stopping criterion: if the change in loss is < tolerance, stop.
        verbose : bool
            If True, prints loss at every 100 iterations.
        """
        self.lr = learning_rate
        self.n_iter = n_iter
        self.tol = tolerance
        self.verbose = verbose

        # placeholders that will be filled during training
        self.w_ = None          # weights (including bias as w[0])
        self.loss_history_ = [] # loss at each iteration
        self.X_ = None          # feature matrix (after standardisation)
        self.y_ = None          # target vector (0/1)

    # ------------------------------------------------------------------
    # 2. Sigmoid helper (vectorised)
    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    # ------------------------------------------------------------------
    # 3. Cost function (cross‑entropy)
    # ------------------------------------------------------------------
    @staticmethod
    def _cost(y: np.ndarray, p: np.ndarray) -> float:
        # avoid log(0) by clipping
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    # ------------------------------------------------------------------
    # 4. Data preparation – this is where we split X / y, scale, etc.
    # ------------------------------------------------------------------
    def prepare(self, df: pd.DataFrame, target_col: str = 'Diagnosis') -> None:
        """
        Splits `df` into X and y, standardises X (mean=0, std=1),
        and stores the result in the class attributes.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned data – *already* contains a numeric target in `target_col`.
        target_col : str
            Name of the binary target column.
        """
        # target must be a 0/1 array
        self.y_ = df[target_col].values.astype(np.int64)

        # X – all columns except the target
        X_raw = df.drop(columns=[target_col]).values.astype(np.float64)

        # -----------------------------------------------------------------
        # 3.1  Feature scaling – we put the bias in the first column
        # -----------------------------------------------------------------
        # compute mean / std on the whole training set (no train/val split yet)
        self.mean_ = X_raw.mean(axis=0)
        self.std_ = X_raw.std(axis=0)
        # avoid division by zero
        self.std_[self.std_ == 0] = 1.0

        X_scaled = (X_raw - self.mean_) / self.std_
        # add bias column (all ones)
        X_scaled = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])

        self.X_ = X_scaled
        self.w_ = np.zeros(X_scaled.shape[1])  # initialise weights

    # ------------------------------------------------------------------
    # 4. Fit – batch gradient descent
    # ------------------------------------------------------------------
    def fit(self) -> None:
        """Runs batch gradient descent for `n_iter` epochs."""
        for i in range(1, self.n_iter + 1):
            z = np.dot(self.X_, self.w_)          # linear part
            p = self._sigmoid(z)                   # predicted probabilities

            # gradient of the log‑likelihood (including bias)
            gradient = np.dot(self.X_.T, (p - self.y_)) / self.y_.size

            # weight update
            self.w_ -= self.lr * gradient

            # record cost and check stopping criterion
            loss = self._cost(self.y_, p)
            self.loss_history_.append(loss)

            if self.verbose and i % 100 == 0:
                print(f"Iteration {i:4d} – loss: {loss:.6f}")

            if i > 1 and abs(self.loss_history_[-2] - loss) < self.tol:
                if self.verbose:
                    print(f"Converged after {i} iterations.")
                break

    # ------------------------------------------------------------------
    # 5. Predict – binary class labels
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 0/1 predictions for a new X matrix (already scaled)."""
        z = np.dot(X, self.w_)
        probs = self._sigmoid(z)
        return (probs >= 0.5).astype(int)

    # ------------------------------------------------------------------
    # 6. Score – accuracy on a given (X, y) pair
    # ------------------------------------------------------------------
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the classification accuracy."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


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

    df['Diagnosis'] = df['Diagnosis'].map({'M': 1, 'B': 0})  # making diagnosis numeric
    df['Diagnosis'] = df['Diagnosis'].astype('category')

    # ---- 7.2  Instantiate and train ------------------------------------
    model = LogisticRegressionGD(learning_rate=0.05,
                                 n_iter=5000,
                                 tolerance=1e-6,
                                 verbose=True)

    # we need to split X / y here
    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis'].cat.codes.values   # 0/1 array

    # Standardise X inside the model for us – we’ll do it in `prepare`
    model.X_ = (X - X.mean()) / X.std()          # bias‑column will be added later
    model.X_ = np.hstack([np.ones((model.X_.shape[0], 1)), model.X_])  # add bias
    model.y_ = y

    # Fit the model
    model.fit()

    # -------------------------------------------------
    # 8. Evaluate on the same data (you could split)
    # -------------------------------------------------
    acc = model.score(model.X_, model.y_)
    print(f"Training accuracy (on the whole cleaned set): {acc:.4f}")

    # Example: predict on the first 10 samples
    y_hat = model.predict(model.X_[:10])
    print("First 10 predictions:", y_hat)

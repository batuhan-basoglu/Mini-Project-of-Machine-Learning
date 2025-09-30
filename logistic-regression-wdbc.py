import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score


class LogisticRegression:
    '''
        Constructor for the logistic regression with gradient descent. It uses learning rate, iteration number,
        tolerance and verbose. It also initializes the weight, loss, x, y, mean and std.
    '''

    def __init__(self, learning_rate: float, n_iter: int, tolerance: float, verbose: bool) -> None: # add momentum as value for the gradient descent
        self.lr = learning_rate
        self.n_iter = n_iter
        self.tol = tolerance
        self.verbose = verbose
        #self.momentum = momentum  # momentum parameter
        self.w: np.ndarray | None = None         # weight/coefficient (bias as first element)
        self.loss: list[float] = []              # loss per iteration
        self.x: np.ndarray | None = None         # matrix of inputs after standardisation
        self.y: np.ndarray | None = None         # target vector
        self.mean: np.ndarray | None = None      # used for standardisation
        self.std: np.ndarray | None = None       # standard deviation
        #self.v: np.ndarray | None = None         # velocity term for momentum

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
            raise RuntimeError("Model is not fitted yet. Call `fit` first.")

        #self.v = np.zeros_like(self.w) # initiating the velocity

        for i in range(1, self.n_iter + 1):
            z = self.x.dot(self.w) # linear prediction
            p = self.sigmoid(z) # probabilities of the model predictions

            gradient = self.x.T.dot(p - self.y) / self.y.size  # for logistic regression X^T*(p - y)

            #self.v = self.momentum * self.v + gradient # incorporating momentum
            #self.w -= self.lr * self.v
            self.w -= self.lr * gradient # gradient multiplied by learning rate is removed from weight

            loss = self.cost(self.y, p) # cost is calculated through cross‑entropy and added for the current range
            self.loss.append(loss)

            # if verbose, it shows the loss every 100 iterations and displays it
            if self.verbose and i % 100 == 0:
                precision = self.precision(self.x, self.y)
                recall = self.recall(self.x, self.y)
                f1_score = self.f1_score(self.x, self.y)
                # 'au_roc = self.au_roc(self.x, self.y)
                print(f"Iter {i:4d} – loss: {loss:.6f} | precision: {precision:.6f} | recall: {recall:.6f} | f1_score: {f1_score:.6f}")

            # tests whether the absolute change in loss is smaller than the tolerance
            if i > 1 and abs(self.loss[-2] - loss) < self.tol:
                if self.verbose:
                    print(f"Converged after {i} iterations.")
                break # loss is stopped so further training would be unnecessary

    def predict(self, x: pd.DataFrame) -> np.ndarray:
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

    def score(self, x: pd.DataFrame, y: pd.Series) -> float:
        """
            This method is used to calculate mean accuracy with the prediction of Y and actual Y values.
        """
        y_pred = self.predict(x)
        y_true = np.asarray(y).astype(int)
        return np.mean(y_pred == y_true) # mean is calculated if Y values match

    def confusion_matrix(self, x: pd.DataFrame, y: pd.Series,
                         normalize: bool = False) -> np.ndarray:
        """
        Confusion Matrix
        Returns a 2x2 matrix: [[TN, FP], [FN, TP]]
        """
        y_pred = self.predict(x)
        y_true = np.asarray(y).astype(int)

        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        return cm

    def plot_confusion_matrix(self, x: pd.DataFrame, y: pd.Series,
                              normalize: bool = False, title: str = "Confusion Matrix", sns=None) -> None:
        """
        Plot confusion matrix as a heatmap
        """
        cm = self.confusion_matrix(x, y, normalize)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                    cmap='Blues', cbar=False,
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def precision(self, x: pd.DataFrame, y: pd.Series) -> float:
        """
        Precision = TP / (TP + FP)
        Measures how many of the predicted positives are actually positive
        """
        cm = self.confusion_matrix(x, y)
        tp, fp = cm[1, 1], cm[0, 1]

        if tp + fp == 0: #div by 0!!!
            return 0.0

        return tp / (tp + fp)

    def recall(self, x: pd.DataFrame, y: pd.Series) -> float:
        """
        Recall = TP / (TP + FN)
        ratio of true positives to all the positives in ground truth
        """
        cm = self.confusion_matrix(x, y)
        tp, fn = cm[1, 1], cm[1, 0]

        if tp + fn == 0:
            return 0.0  # Avoid division by zero

        return tp / (tp + fn)

    def f1_score(self, x: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> float:
        """
        F1-Score = 2 * ((Precision * Recall) / (Precision + Recall))
        """
        prec = self.precision(x, y)
        rec = self.recall(x, y)

        if prec + rec == 0:
            return 0.0  # Avoid division by zero

        return 2 * ((prec * rec) / (prec + rec))

    '''
    def predict_proba(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predict probability scores instead of binary labels
        """
        if isinstance(x, pd.DataFrame):
            x = x.values

        if self.w is None:
            raise ValueError("Model not fitted yet")

        # Add bias term if needed
        if x.shape[1] == len(self.w) - 1:
            x = np.column_stack([np.ones(x.shape[0]), x])

        return self.sigmoid(x @ self.w)
    def au_roc(self, x: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> float:
        """
        Measures the model's ability to distinguish between classes
        """
        # make sure self.sigmoid outputs floats between 0 and 1
        y_true = np.asarray(y).astype(int)
        y_proba = self.predict_proba(x)

        return roc_auc_score(y_true, y_proba)
'''
    def classification_report(self, x: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> dict:
        """
        Comprehensive classification report
        """
        return {
            'precision': self.precision(x, y),
            'recall': self.recall(x, y),
            'f1_score': self.f1_score(x, y),
            #'au_roc': self.au_roc(x, y)
        }

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

    # ID should be dropped --> remove 1st row
    df.drop(columns=['ID'], inplace=True) # drops id column

    # no duplicate rows but just in case:
    df = df.drop_duplicates()
    # check data types: --> everything is good
    # print(df.dtypes)
    '''
    # ____________________________________________________________________________________
    # HANDLE OUTLIERS AND INCONSISTENCIES
    # https://medium.com/@heyamit10/pandas-outlier-detection-techniques-e9afece3d9e3
    # if z-score more than 3 --> outllier
    # print(cancer.head().to_string())

    # ____________________________________________________________________________________

    # separate dependent VS independent variables
    x = cancer.drop(cancer.columns[0], axis=1)
    y = cancer[1]

    # print(x.head().to_string())

    # normalize data
    # normalize = cancer.drop(cancer.columns[0], axis=1)
    # normalize = (normalize - normalize.mean()) / normalize.std()
    # cancer[cancer.columns[1:]] = normalize
    # print(cancer.head().to_string())

    # turn into array for regression
    x = x.to_numpy()
    y = y.to_numpy()

    # cancer_y = np.asarray(cancer2[0].tolist())
    # cancer2.drop(cancer2[0], axis = 1, inplace = True)

    # split data into train / tests datasets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
'''
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

    #check for correlation radius, are and perimeter have trivially a high correlation
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # find features with correlation greater than 0.90
    high_corr_features = []
    for col in upper.columns:
        high_corr = upper[col][upper[col] > 0.90]
        if not high_corr.empty:
            high_corr_features.append((col, high_corr.index.tolist()))

    if high_corr_features:
        print("correlated features (>0.95):")
        for feature, correlated_with in high_corr_features:
            print(f"  {feature} AND {correlated_with}")

        # check for weak correlation with target --> worsts have the most impact
        target_corr = df.corr()['Diagnosis'].abs().sort_values(ascending=False)
        print("\nCorrelation with target variable descending order:")
        print(target_corr)
    print("") # \n splitter

    rng = np.random.default_rng(seed=42)
    n_train = len(df)
    indices = rng.permutation(n_train)
    train_size = int(0.8 * n_train)

    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # training of the model
    model = LogisticRegression(learning_rate=0.00005, n_iter=5000, tolerance=1e-6, verbose=True)
    # other values could be used, for example (lr=0.01, n_iter=2000, tolerance=1e-3, verbose=False)
    #model = LogisticRegression(learning_rate=0.00005, n_iter=5000, tolerance=1e-6, verbose=True, momentum= 0.9)
    # using momentum for gradient descent calculation
    model.prepare(df_train, target_col="Diagnosis")
    model.fit()

    # evaluation of the model
    train_acc = model.score(model.x, model.y)
    print(f"\nMean accuracy on training data: {train_acc:.4f}")

    # copied prepare method for building test X data
    x_test_raw = df_test.drop(columns=['Diagnosis']).values.astype(np.float64)
    x_test_scaled = (x_test_raw - model.mean) / model.std
    bias_test = np.ones((x_test_scaled.shape[0], 1), dtype=np.float64)
    x_test = np.hstack((bias_test, x_test_scaled))
    y_test = df_test['Diagnosis'].values.astype(int)
    test_acc = model.score(x_test, y_test)
    print(f"Mean accuracy on testing data: {test_acc:.4f}")

    # predict Y values using the trained data
    first_10 = x_test[:10]
    y_hat = model.predict(first_10)
    print("\nFirst 10 predictions:", y_hat.ravel())

    # weight report
    print("\nWeights from the model:")
    print(model.w)
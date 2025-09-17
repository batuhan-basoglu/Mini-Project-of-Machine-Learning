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
    



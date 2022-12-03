from sklearn.pipeline import TransformerMixin

class FilterOutBigValuesTransformer(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.biggest_value = X.c1.max()
        return self

    def transform(self, X):
        return X.loc[X.c1 <= self.biggest_value]
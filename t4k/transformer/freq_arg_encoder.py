from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FreqArgEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encode_dicts = None

    def fit(self, X,y=None):
        """
        Create a map of category -> frequency for each column.
        Unknown is set to -1
        
        Parameters
        ----------
        X : Pandas.DataFrame
            Training data.
        y : None
            None
            
        Returns
        -------
        Trained encoder : self
            self.
        """
        self.encode_dicts = [X.iloc[:,i].value_counts() for i in range(X.shape[1])]
        for each_dict in self.encode_dicts:
            which_type = np.int8 if len(each_dict) < 128 \
                else np.int16 if len(each_dict) < pow(2,15) \
                else np.int32 if len(each_dict) < pow(2,31) \
                else np.int64
            each_dict[:] = np.arange(len(each_dict)).astype(which_type)
        return self
    
    def transform(self, X):
        ret_X = X.copy()
        for col_id in range(X.shape[1]):
            ret_X.iloc[:,col_id] = X.iloc[:,col_id].map(self.encode_dicts[col_id])
        ret_X.fillna(-1, inplace=True)
        return ret_X
    
    def predict(self, X):
        return X

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class VersionEncoder (BaseEstimator, TransformerMixin):
    """
    Encode version column s.a categoy 4.12.0.128 to int 4120128
    Example
    -------
    input DataFrame:
        version1    version2
        4.100.8     1.0.0.1
        4.200.16    2.13.0.1

    output DataFrame:
        version1    version2
        410008      10001
        420016      21301
    """
    
    def __init__(self):
        self.version_depth_dict = None # e.g. 4 if version "4.12.0.18"
        self.version_len_dict = None # e.g. np.array([1,2,1,2]) if version "4.12.0.18"

    def fit(self, X,y=None):
        """
        Create encoder Series for each column.
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

        Example
        -------
        input DataFrame:
            version1    version2
            4.100.8     1.0.0.1
            4.200.16    2.13.0.1

        self.version_depth_dict:
            { 'version1': 3,
              'version2': 4 }
        self.version_len_dict:
            { 'version1': pd.Series([ 2, 3,2]),
              'version2': pd.Series([ 1,2,1,1]) }
        """

        col_names = X.columns
        self.version_depth_dict = {col_name : X[col_name].apply(lambda x:len(x.split('.'))).max() \
            for col_name in col_names}

        version_df_dict = {col_name: pd.DataFrame({'{0}_{1}'.format(col_name,i): \
            X[col_name].map(lambda x:x.split('.')[i]) \
            for i in range(self.version_depth_dict[col_name])}) for col_name in col_names}

        self.version_len_dict = { col_name: version_df_dict[col_name] \
            .applymap(lambda x:len(x)).max() for col_name in col_names }
        return self

    def transform(self, X):
        col_names = X.columns
        version_df_dict = {col_name: pd.DataFrame({'{0}_{1}'.format(col_name,i): \
            X[col_name].map(lambda x:x.split('.')[i]) \
            for i in range(self.version_depth_dict[col_name])}) for col_name in col_names}

        for col_name in col_names:
            for i in range(self.version_depth_dict[col_name]):
                size = self.version_len_dict[col_name][i]
                version_df_dict[col_name].iloc[:,i] = version_df_dict[col_name].iloc[:,i]\
                    .map(lambda x: x.zfill(size)[:size])

            version_df_dict[col_name] = version_df_dict[col_name].sum(axis=1)
        unioned = pd.DataFrame({col_name: version_df_dict[col_name] for col_name in col_names}) 
        return unioned

    def predict(self, X):
        return X

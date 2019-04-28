from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders as ce
import numpy as np
import pandas as pd
from tqdm import tqdm

class TimeSeriesTargetEncoder(BaseEstimator, TransformerMixin):
    """
        Encode category to target value frequency on past data.

        Parameters
        ----------
        cols: list
            a list of columns to encode, if None, all columns will be encoded
        time_col: str
            a name of time series column used for identity the time of that row
        handle_unknown: float
            a value used for unknown category
    """
    def __init__(self,cols=None, time_col=None, handle_unknown=None, valid_appearance=0.05, ignore_first=0.1):
        self.cols = cols
        self.time_col = time_col
        self.handle_unknown = handle_unknown
        self.valid_appearance = valid_appearance
        self.ignore_first = ignore_first


    def fit(self, X, y):
        """
        Encode category data.
        
        Parameters
        ----------
        X : Pandas.DataFrame
            Training data.
        y : None
            None
            
        Returns
        -------
        self : encoder
            Returns self.
        """
        order_series = X[self.time_col]
        features = X[self.cols]
        # filter_rare category
        self.valid_category_dict = {}
        for col in self.cols:
            freq = features[col].value_counts() / features.shape[0]
            self.valid_category_dict[col] = freq[freq > self.valid_appearance].index.values

        # sort by order column
        ordered = features.assign(order_series=order_series) \
            .assign(target_series=y).sort_values('order_series') \
            .apply(lambda col:col.cat.add_categories(['na']) if col.dtype.name=='category' else col) \
            .fillna('na')
        ordered_y = ordered['target_series']
        ohe = ce.OneHotEncoder(cols=self.cols, use_cat_names=True, handle_unknown='ignore')
        one_hot_encoded = ohe.fit_transform(ordered.drop(['target_series'],axis=1))

        time_incre_count = one_hot_encoded.cumsum()
        one_hot_target = one_hot_encoded * np.repeat(ordered_y[:,np.newaxis], one_hot_encoded.shape[1], axis=1)
        time_incre_target = one_hot_target.cumsum()

        te_table_all = (time_incre_target / time_incre_count).drop(['order_series'],axis=1)
        te_table_all['order_series'] = one_hot_encoded['order_series']
        self.te_table = te_table_all.groupby(by='order_series').mean()
        self.te_table['unknown_category'] = np.nan
        self.te_table_columns = list(self.te_table.columns)
        return self

    def transform(self, X):
        order_series = X[self.time_col]
        features = X[self.cols]

        # filter rare appearance
        features = features.apply(lambda column: column.apply(\
            lambda x:x if x in self.valid_category_dict[column.name] else 'na_test'
            ), axis=0)

        ordered = features.assign(order_series=order_series).sort_values('order_series') \
            .apply(lambda col:col.cat.add_categories(['na_test']) if col.dtype.name=='category' else col) \
            .fillna('na_test')
        result_list = []

        def convert_for_te(row):
            concat_col_name = [col + '_' + str(val) for col, val in zip(self.cols, row)]
            return [elem if elem in self.te_table_columns else 'unknown_category' for elem in concat_col_name]

        te_colname_list_array = ordered[self.cols].apply(convert_for_te, axis=1).values
        order_array = ordered.order_series.values

        # search corresponded te_table index
        corresponded_id = np.searchsorted(self.te_table.index.values, order_array, side='left')
        corresponded_id = corresponded_id - 1
        corresponded_id = np.clip(corresponded_id, a_min=0, a_max=None)
        corresponded_order = self.te_table.index.values[corresponded_id]

        # correspond index and cols df
        corresponded_df = pd.DataFrame({
            'corr_order':corresponded_order,
            'corr_te_cols':te_colname_list_array
        })

        corresponded_df = corresponded_df.apply(lambda row:\
            pd.Series(self.te_table.loc[row['corr_order'],row['corr_te_cols']].values, index=self.cols),
            axis=1)

        corresponded_df.index = ordered.index
        
        # ignore earlier time row
        threshold_version = self.te_table.index[int(self.ignore_first * self.te_table.shape[0])]
        corresponded_df[order_array < threshold_version] = np.nan
        return corresponded_df.sort_index()
    
    def predict(self, X):
        return X

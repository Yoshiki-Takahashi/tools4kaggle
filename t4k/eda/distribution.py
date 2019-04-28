import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def my_print():
    print('hello')

def train_test_dist(train_df, test_df, col_name,
        target_name=None,
        fig_size=(10,4),
        disp_range=1.0,
        disp_num=None,
        bins=None):

    fig, ax = plt.subplots(figsize=fig_size)

    def numeric_dist():
        min_max = test_df[col_name].quantile([(1-disp_range)/2, (1+disp_range)/2])
        sns.distplot(
            train_df[train_df[col_name].between(min_max.iloc[0], min_max.iloc[1])] \
            .loc[:,col_name], bins=bins, label='train',ax=ax)
        sns.distplot(
            test_df[test_df[col_name].between(min_max.iloc[0], min_max.iloc[1])] \
            .loc[:,col_name], bins=bins, label='test',ax=ax)
        plt.legend()

    def categ_dist():
        xlim = disp_num or train_df[col_name].nunique()
        train_count = (train_df[col_name].value_counts() / len(train_df)).to_frame()\
            .assign(data=lambda x:'train')\
            .reset_index()\
            .rename(columns={col_name:'count','index':col_name}) \
            .iloc[:xlim,:]
        test_count = (test_df[col_name].value_counts() / len(test_df)).to_frame()\
            .assign(data=lambda x:'test')\
            .reset_index()\
            .rename(columns={col_name:'count','index':col_name})
        train_test_count = pd.concat([train_count, test_count])
        sns.barplot(x=col_name, y='count', hue='data', order=train_count[col_name], data=train_test_count, alpha=0.5, ax=ax)
        if not target_name == None:
            is_binary_target = train_df[target_name].nunique() == 2

            target_ratio = train_df.loc[:,[col_name,target_name]].groupby(col_name).mean()\
                .loc[train_count[col_name],:] # Sort same order to histogram.
            ax2 = ax.twinx()
            target_ratio.plot.line(ax=ax2, color='r')
            ax2.set_ylabel(target_name + ' mean')
            if is_binary_target:
                ax2.set_ylim(0,1)
        if not disp_num == None:
            ax.set_xlim(-0.5, disp_num)
        for tick in ax.get_xticklabels():
            tick.set_rotation(-90)

    col_type = str(train_df[col_name].dtype)
    if 'float' in col_type or 'int' in col_type:
        numeric_dist()
    elif col_type == 'category':
        categ_dist()
    else:
        print('Not supported dtype')

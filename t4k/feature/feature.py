# This code comes from https://amalog.hateblo.jp/entry/kaggle-feature-management
# Copy Rights belongs to えじ (amaotone).

import re
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path
from contextlib import contextmanager

import pandas as pd

@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = '.'
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = Path(self.dir) / f'{self.name}_train.ftr'
        self.y_train_path = Path(self.dir) / f'{self.name}_y_train.ftr'
        self.test_path = Path(self.dir) / f'{self.name}_test.ftr'
    
    def run(self):
        with timer(self.name):
            self.create_features()
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self
    
    @abstractmethod
    def create_features(self):
        raise NotImplementedError
    
    def save(self):
        self.train.to_feather(str(self.train_path))
        self.y_train.to_feather(str(self.y_train_path))
        self.test.to_feather(str(self.test_path))


def load_datasets(kernel_title, feats):
    dfs = [pd.read_feather(f'../input/{kernel_title}/{f}_train.ftr') for f in feats]
    X_train = pd.concat(dfs, axis=1)
    first_f = feats[0]
    y_train = pd.read_feather(f'../input/{kernel_title}/{first_f}_y_train.ftr')
    dfs = [pd.read_feather(f'../input/{kernel_title}/{f}_test.ftr') for f in feats]
    X_test = pd.concat(dfs, axis=1)
    return X_train, y_train, X_test

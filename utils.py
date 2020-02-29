from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

from import_prepare_data_sets import (
    get_two_features_data,
    get_two_features_plus_noise_data
)

def get_top_two_features(X_transformed, y_labels, num_cols):
    rf = RandomForestClassifier()
    rf.fit(X_transformed, y_labels)
    feature_importances = pd.DataFrame({'importances': rf.feature_importances_})
    feature_importances = feature_importances.iloc[0:len(num_cols), :]
    feature_importances.index = num_cols
    top_two = feature_importances.sort_values(ascending=False,
                                              by='importances')[0:2].index
    top_two_idx = []
    for i in range(2):
        top_two_idx.append(int(np.where(np.array(num_cols) == top_two[i])[0]))
    return top_two, top_two_idx


def save_data(X_train, X_test, y_train, y_test, name):
    X_train.to_csv('./splits/' + name + '_X_train.csv', index=False)
    X_test.to_csv('./splits/' + name + '_X_test.csv', index=False)
    y_train.to_csv('./splits/' + name + '_y_train.csv', header=True, index=False)
    y_test.to_csv('./splits/' + name + '_y_test.csv', header=True, index=False)


def load_data(name):
    X_train = pd.read_csv('./splits/' + name + '_X_train.csv')
    X_test = pd.read_csv('./splits/' + name + '_X_test.csv')
    y_train = pd.read_csv('./splits/' + name + '_y_train.csv')
    y_test = pd.read_csv('./splits/' + name + '_y_test.csv')
    return X_train, X_test, y_train, y_test


def build_pipeline(categorical_features, numeric_features, clf=None):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    if clf is not None:
        pipe = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', clf)])
        return pipe
    else:
        return preprocessor

def make_grid_search_cv(pair, cat_cols, num_cols, scoring, objective, n_jobs):
    """Return grid search cv with pipeline, model,
    and hyperparameter grid
    """
    #print(pair)
    model, hyperparams = pair
    if model.__name__ == 'SVC':
        model_init = model(probability=True)
    else:
        model_init = model()
    pipe = build_pipeline(cat_cols, num_cols, model_init)
    gridCV = GridSearchCV(pipe,
                          hyperparams,
                          scoring=scoring,
                          refit=objective,
                          cv=5,
                          n_jobs=n_jobs,
                          return_train_score=True
                          )
    return gridCV


def get_data_set_imports(sets):
    funcs = [
        ('two_features', get_two_features_data),
        ('data_with_noise', get_two_features_plus_noise_data)
    
    ]
    return funcs


def strip_prefix_from_dict(params, letters=7):
    return {k[letters:]: v for k, v in params.items()}

def file_stem(name, model_name):
    return name + '_' + model_name


def file_join(l):
    l = [str(item) for item in l]
    return '_'.join(l)


def log_filename(problems, STEM):
    fn = str(datetime.now())
    probs = '_'.join(problems)
    return STEM + '_' + fn + '_' + probs





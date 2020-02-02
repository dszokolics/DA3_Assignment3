import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso


# I would do cross validation based on the boroughs, instead of random splitting.
def k_fold_brg(abt, feat, model, y_var, neigh, method='all', log=False, brgs=None, valid=None):

    abt = abt.copy()
    feat = feat.copy()

    np.random.seed(12)

    if brgs is None:
        brgs = abt.neighbourhood_cleansed.drop_duplicates().sort_values().tolist()
        brgs = [x for x in brgs if x != 'Hackney']

    rmse_test = []
    rmse_train = []
    r2_test = []
    r2_train = []
    stats = {}

    if method in ['all', 'io']:
        feat.append('neigh')

    for brg in brgs:

        # Drop one of the neighbourhood features to avoid linear dependence for linear models
        if isinstance(model, LinearRegression) | isinstance(model, Lasso):
            feat = [x for x in feat if x not in [brg, [x for x in brgs if x != brg][0]]]

        if method in ['all', 'io']:
            abt['neigh'] = abt.neighbourhood_cleansed.isin(neigh[brg])

        if valid is not None:
            X_test = valid.loc[valid.neighbourhood_cleansed == brg, feat].copy()
            y_test = valid.loc[valid.neighbourhood_cleansed == brg, y_var].copy()
        else:
            X_test = abt.loc[abt.neighbourhood_cleansed == brg, feat].copy()
            y_test = abt.loc[abt.neighbourhood_cleansed == brg, y_var].copy()

        if method == 'all':
            X_train = abt.loc[abt.neighbourhood_cleansed != brg, feat].copy()
            y_train = abt.loc[abt.neighbourhood_cleansed != brg, y_var].copy()

        elif method == 'io':
            inner = abt.loc[abt.neighbourhood_cleansed == brg, 'inner_district'].values[0]
            X_train = abt.loc[(abt.neighbourhood_cleansed != brg)
                              & (abt.inner_district == inner), feat].copy()
            y_train = abt.loc[(abt.neighbourhood_cleansed != brg)
                              & (abt.inner_district == inner), y_var].copy()

        elif method == 'neigh':
            X_train = abt.loc[(abt.neighbourhood_cleansed).isin(neigh[brg]),
                              feat].copy()
            y_train = abt.loc[(abt.neighbourhood_cleansed).isin(neigh[brg]),
                              y_var].copy()

        elif method == 'io_neigh':
            inner = abt.loc[abt.neighbourhood_cleansed == brg, 'inner_district'].values[0]
            X_train = abt.loc[(abt.neighbourhood_cleansed).isin(neigh[brg])
                              | (abt.inner_district == inner)
                              & (abt.neighbourhood_cleansed != brg),
                              feat].copy()
            y_train = abt.loc[(abt.neighbourhood_cleansed).isin(neigh[brg])
                              | (abt.inner_district == inner)
                              & (abt.neighbourhood_cleansed != brg),
                              y_var].copy()

        else:
            raise ValueError("method can be one of [None, 'all', 'io', 'neigh', 'io_neigh']")

        if log:
            y_orig = y_train.copy()
            y_train = np.log(y_train)

        if isinstance(model, LinearRegression) | isinstance(model, Lasso):
            # Drop the superborough in which the borough is, and all other which are not present in the data
            superbrgs = ['inner_east', 'inner_west', 'outer_w_nw', 'outer_e_ne', 'outer_south']
            superbrgs = X_train[superbrgs].nunique()
            to_drop = superbrgs[superbrgs == 1].index.tolist() + [superbrgs[superbrgs > 1].index.values[0]]
            X_train.drop(columns=to_drop, inplace=True)
            X_test.drop(columns=to_drop, inplace=True)

        model.fit(X_train, y_train)

        if log:
            y_train_hat = model.predict(X_train)
            y_test_hat = model.predict(X_test)
            var_hat = (y_train - y_train_hat).var()

            y_train_hat = np.exp(y_train_hat) * np.exp(var_hat/2)
            y_test_hat = np.exp(y_test_hat) * np.exp(var_hat/2)

            mse_train = mean_squared_error(y_orig, y_train_hat)
            mse_test = mean_squared_error(y_test, y_test_hat)

            rmse_train.append(np.sqrt(mse_train))
            rmse_test.append(np.sqrt(mse_test))
            r2_train.append(1 - mse_train / y_orig.var())
            r2_test.append(1 - mse_test / y_test.var())

        else:
            rmse_train.append(np.sqrt(mean_squared_error(y_train, model.predict(X_train))))
            rmse_test.append(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))
            r2_train.append(model.score(X_train, y_train))
            r2_test.append(model.score(X_test, y_test))

        if isinstance(model, RandomForestRegressor):
            stats[brg] = model.feature_importances_
        elif isinstance(model, LinearRegression) | isinstance(model, Lasso):
            i = 0
            stat = []
            for col in feat:
                if col in X_train.columns:
                    stat.append(model.coef_[i])
                    i += 1
                else:
                    stat.append(0)
            stats[brg] = stat

    res = pd.DataFrame({'brg': brgs,
                        'rmse_train': rmse_train,
                        'rmse_test': rmse_test,
                        'r2_train': r2_train,
                        'r2_test': r2_test})

    stats = pd.DataFrame(stats).transpose()
    stats.columns = feat

    return res, stats

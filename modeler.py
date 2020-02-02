import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
from tqdm import tqdm
import datetime as dt
import pickle
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

from utils import cleaner
from utils.helpers import k_fold_brg
from utils.feature_engineering import add_new_features


def avg_weighted(res, weights, idx=None):
    """Weights the results among the different boroughs, and creates a summary
    on the weighted performance."""

    if idx is None:
        rrr = res.copy()
        rrr.rename(columns={'borough': 'brg'}, inplace=True)
    else:
        rrr = res[idx].copy()

    if idx == 1:
        rrr.reset_index(inplace=True)
        rrr.rename(columns={'index': 'brg'}, inplace=True)

    feat_num = rrr.shape[1] - 1
    rrr = pd.merge(rrr, brgs.rename(columns={'inner_district': 'i_d'}),
                   how='left', left_on='brg', right_on='borough')
    inner_east = ['Haringey', 'Islington', 'Lambeth', 'Lewisham',
                  'Newham', 'Southwark', 'Tower Hamlets']
    rrr['i_e'] = rrr.brg.isin(inner_east)
    rrr['near_hackney'] = rrr.brg.isin(br_neigs['Hackney'])
    rrr['weight'] = (1
                     + rrr.i_d * weights['inner']
                     + rrr.i_e * weights['inner_east']
                     + rrr.near_hackney * weights['neighbor'])
    indicators = rrr.columns[1:feat_num+1].tolist()
    for col in indicators:
        rrr[f'w_{col}'] = rrr[col] * rrr.weight
    rrr = rrr.sum()
    round_num = 3 if idx == 0 else 4
    for i in ['w_'+x for x in indicators]:
        rrr[i] = round((rrr[i] / rrr['weight']), round_num)

    rrr = rrr[['w_'+x for x in indicators]]
    if idx == 1:
        rrr.sort_values(inplace=True, ascending=False)
        rrr.index = [x[2:] for x in rrr.index]

    return rrr


def eval_models(res, weights):
    """Aggregates different performance measures into a single output."""

    all_res_weighted_mean = pd.concat([avg_weighted(x, weights, 0) for x in res], axis=1)
    all_res_weighted_mean.columns = [f'({x})' for x in range(0, len(res))]
    all_res_weighted_mean.astype(float).round(3).sort_index(ascending=False)

    return all_res_weighted_mean


# Read data
ab_raw = pd.read_csv('data/airbnb_london_cleaned.csv', index_col=0)
brgs = pd.read_csv('data/london_boroughs.csv')
br_neigs = pickle.load(open('data/br_neigs.pickle', 'rb'))

# Clean the data (using helper functions from the cleaner module)
brgs = cleaner.clean_borough_data(brgs)
ab = cleaner.clean_airbnb_data(ab_raw)
ab, validation, hackney = cleaner.split_airbnb_data(ab)
ab, validation, hackney = cleaner.clean_abt_parts(ab, validation, hackney, brgs)

ab_raw.shape
ab.shape

# Dependent variable
y_var = 'price'

# Independent variables
X_feat = (ab.select_dtypes('number').columns.tolist()
    + ab.select_dtypes('bool').columns.tolist())
not_X = ['id', y_var]
X_feat = [x for x in X_feat if x not in not_X]
len(X_feat)

# Drop variables with zero Standard Deviation
stds = ab[X_feat].std()
X_feat = stds[stds > 0].index.tolist()

ab2 = ab[X_feat + [y_var, 'neighbourhood_cleansed']].copy()

# Filtering the dataset (only train+test)
# Based on general economic trends, apartments above 1000 price are very unlikely
#   in hackney
ab2[ab2.price > 1000].groupby('neighbourhood_cleansed').accommodates.count()
ab2.loc[ab2.price > 1000, ['neighbourhood_cleansed', 'price']]

ab2 = ab2[ab2.price <= 1000]

### Random Forest
rfr = RandomForestRegressor(max_features='sqrt', n_estimators=500, n_jobs=2)

# Try out all four different train set specifications
res_all = k_fold_brg(ab2, X_feat, rfr, y_var, neigh=br_neigs)
res_all[0]

res_io = k_fold_brg(ab2, X_feat, rfr, y_var, neigh=br_neigs, method='io')
res_io[0]

res_neigh = k_fold_brg(ab2, X_feat, rfr, y_var, method='neigh', neigh=br_neigs)
res_neigh[0]

res_io_neigh = k_fold_brg(ab2, X_feat, rfr, y_var, method='io_neigh', neigh=br_neigs)
res_io_neigh[0]

# Evaluate
all_res = [res_all, res_io, res_neigh, res_io_neigh]
all_res_simple_mean = pd.concat([x[0].mean() for x in all_res], axis=1)
all_res_simple_mean.columns = ['(1)', '(2)', '(3)', '(4)']
all_res_simple_mean.astype(float).round(3).sort_index(ascending=False)

# Specify weight for boroughs
w1 = {'inner': 0.4, 'inner_east': 0.1, 'neighbor': 0.2}

# Get the weighted results
all_res = [res_all, res_io, res_neigh, res_io_neigh]
eval_models(all_res, w1)

# Updated weights based on the results
w2 = {'inner': 0.5, 'inner_east': 0.2, 'neighbor': 0.5}
rfr2 = RandomForestRegressor(max_features='sqrt', n_estimators=1000, n_jobs=2)
res_io_neigh_x = k_fold_brg(ab2, X_feat, rfr2, y_var, method='io_neigh', neigh=br_neigs)
res_io_neigh_x[0]

avg_weighted(res_io_neigh_x, w2, 0)
fi_mean = avg_weighted(res_io_neigh_x, w2, 1)

# Try out models with filtered feature set
rfr2 = RandomForestRegressor(max_features='sqrt', n_estimators=1000, n_jobs=2)
feat_res = []
for f_rest in tqdm([1, 5, 10, 20, 40, 100]):  # tqdm(range(15, 36, 10)):
    X_feat_2 = fi_mean[fi_mean > f_rest / 10000].index.tolist()
    print(len(X_feat_2))
    feat_res.append(k_fold_brg(ab2, X_feat_2, rfr2, y_var, method='io_neigh', neigh=br_neigs))
eval_models(feat_res, w2)

# Save the features for the best performing setup
X_feat_rfr = fi_mean[fi_mean > 0.001].index.tolist()
len(X_feat_rfr)

# Try out models with different max_depth parameter
depth_res = []
for depth in tqdm(range(12, 22, 2)):
    rfr2 = RandomForestRegressor(max_features='sqrt', n_estimators=500, max_depth=depth, n_jobs=2)
    depth_res.append(k_fold_brg(ab2, X_feat, rfr2, y_var, method='io_neigh', neigh=br_neigs))
eval_models(depth_res, w2)

# Try out models with different parameter for minimum samples needed to split
split_res = []
for split in tqdm(range(3, 10, 2)):
    rfr2 = RandomForestRegressor(max_features='sqrt', n_estimators=500, min_samples_split=split, n_jobs=2)
    split_res.append(k_fold_brg(ab2, X_feat, rfr2, y_var, method='io_neigh', neigh=br_neigs))
eval_models(split_res, w2)

# Try out a model restricted in multiple ways
rfr3 = RandomForestRegressor(max_features='sqrt', n_estimators=500, min_samples_split=3, max_depth=20, n_jobs=2)
mix_rest_res = k_fold_brg(ab2, X_feat, rfr3, y_var, method='io_neigh', neigh=br_neigs)
avg_weighted(mix_rest_res, w2, 0)


### Linear models

# Create a third DataFrame for linear models
ab3 = add_new_features(ab2)

# All the linear features
X_feat_lin = [x for x in ab3.columns if x not in ['neighbourhood_cleansed', 'inner_district', y_var]]

stds_to_scale = ab3[X_feat_lin].std()
means_to_scale = ab3[X_feat_lin].mean()
ab3[X_feat_lin] = preprocessing.scale(ab3[X_feat_lin])

# Lasso regression
metrics = []
coefs = []
for a in tqdm(range(-1, 1)):
    ls = Lasso(alpha=2**a, max_iter=15000)
    ls_res = k_fold_brg(ab3, X_feat_lin, ls, y_var, method='io_neigh', neigh=br_neigs)
    metrics.append(avg_weighted(ls_res, w2, 0))
    coefs.append(avg_weighted(ls_res, w2, 1))

colnames = ([f'{round(2**(x), 3)}_v2' for x in range(-7, -4)]
            + [f'{round(2**-4, 3)}_v2']
            + [f'{round(2**(x), 3)}_v2' for x in range(-3, -1)]
            + [f'{round(2**(x), 3)}_v2' for x in range(-1, 1)])
mets = pd.DataFrame(metrics).transpose()
cfs = pd.DataFrame(coefs).transpose()
mets.columns = colnames
cfs.columns = colnames

# Evaluation
mets.transpose().w_rmse_test.min()
mets

# Feature importances
cfs2 = cfs.copy()
col_x = '0.016_v2'
cfs2['v_abs'] = cfs[col_x].abs()
cfs2[['v_abs', col_x]].sort_values('v_abs', ascending=False).rename(columns={'v_abs': 'Coef_abs', col_x: 'Coef'}).dropna().head(25)

# Linear Regression
lm = LinearRegression()
lm_res = k_fold_brg(ab3, X_feat_lin, lm, y_var, method='io_neigh', neigh=br_neigs, log=False)

avg_weighted(lm_res, w2, 0)
avg_weighted(lm_res, w2, 1)

# Lasso with logarithmic dependent variable
ln_metrics = []
ln_coefs = []
for a in tqdm(range(-17, -13)):
    ls = Lasso(alpha=2**a, max_iter=15000)
    ls_res = k_fold_brg(ab3, X_feat_lin, ls, y_var, method='io_neigh', neigh=br_neigs, log=True)
    ln_metrics.append(avg_weighted(ls_res, w2, 0))
    ln_coefs.append(avg_weighted(ls_res, w2, 1))

ln_colnames = ([f'{round(2**(x), 5)}_v1' for x in range(-17, -13)])
ln_mets = pd.DataFrame(ln_metrics).transpose()
ln_cfs = pd.DataFrame(ln_coefs).transpose()
ln_mets.columns = ln_colnames
ln_cfs.columns = ln_colnames

ln_mets

ln_cfs2 = ln_cfs.copy()
ln_col_x = '6e-05_v1'
ln_cfs2['v_abs'] = ln_cfs2[ln_col_x].abs()
ln_cfs2[['v_abs', ln_col_x]].sort_values('v_abs', ascending=False).rename(columns={'v_abs': 'Coef_abs', ln_col_x: 'Coef'}).dropna().tail(25)

# Linear Regression with logarithmic dependent variable
lm = LinearRegression()
ln_lm_res = k_fold_brg(ab3, X_feat_lin, lm, y_var, method='io_neigh', neigh=br_neigs, log=True)
avg_weighted(ln_lm_res, w2, 0)
avg_weighted(ln_lm_res, w2, 1)


# X_feat_rfr = pickle.load(open('results/X_feat_rfr.pickle', 'rb'))

### Create the set of final models - one model for each borough

# too big to fit into memory - save to disk
for brg in tqdm(ab_raw.neighbourhood_cleansed.unique()):

    if brg == 'Hackney':
        relevant = (set(br_neigs['Hackney'])
                    | set(ab2.loc[ab2.inner_district, 'neighbourhood_cleansed']))
    else:
        inner = ab2.loc[ab2.neighbourhood_cleansed == brg, 'inner_district'].values[0]
        neig = br_neigs[brg]
        relevant = set(neig) | set(ab2.loc[ab2.inner_district == inner, 'neighbourhood_cleansed']) - set(brg)

    final_model = RandomForestRegressor(n_estimators=1000, max_features='sqrt', n_jobs=2)
    final_model.fit(ab2.loc[ab2.neighbourhood_cleansed.isin(relevant), X_feat_rfr],
                    ab2.loc[ab2.neighbourhood_cleansed.isin(relevant), 'price'])

    pickle.dump(final_model, open(f'models/Final_model_{brg}.pickle', 'wb'))


def get_results(data, X_feat):
    """Creates a prediction for each borough in the input data, and returns the
    RMSE, R2 and number of lines for each."""

    res = []
    for brg in tqdm(data.neighbourhood_cleansed.unique()):

        final_model = pickle.load(open(f'models/Final_model_{brg}.pickle', 'rb'))
        subdata = data[data.neighbourhood_cleansed == brg].copy()

        y_pred = final_model.predict(subdata[X_feat])
        rmse = np.sqrt(mean_squared_error(subdata['price'], y_pred))
        r2 = final_model.score(subdata[X_feat], subdata['price'])
        res.append({'borough': brg, 'test_rmse': rmse, 'test_r2': r2, 'N': len(subdata)})

    return pd.DataFrame(res).sort_values('borough')


train_results = get_results(ab2, X_feat_rfr)
train_results.rename(columns={'test_r2': 'train_r2', 'test_rmse': 'train_rmse', 'N': 'train_N'}, inplace=True)
train_results
avg_weighted(train_results, w2)


### Validation
inner = ab2.loc[ab2.neighbourhood_cleansed == 'Camden', 'inner_district'].values[0]
neig = br_neigs['Camden']
relevant = set(neig) | set(ab2.loc[ab2.inner_district == inner, 'neighbourhood_cleansed'])

# final_model = RandomForestRegressor(n_estimators=1000, max_features='sqrt', n_jobs=2)
# final_model.fit(ab2.loc[ab2.neighbourhood_cleansed.isin(relevant), X_feat_rfr],
#             ab2.loc[ab2.neighbourhood_cleansed.isin(relevant), 'price'])

get_results(validation[validation.neighbourhood_cleansed == 'Camden'], X_feat_rfr)
camden_model = pickle.load(open('models/Final_model_Camden.pickle', 'rb'))
camden_result = camden_model.predict(validation.loc[validation.neighbourhood_cleansed == 'Camden', X_feat_rfr])
np.sqrt(mean_squared_error(camden_result, validation.loc[validation.neighbourhood_cleansed == 'Camden', 'price']))


(camden_result - validation.loc[validation.neighbourhood_cleansed == 'Camden', 'price']).hist()

validation.loc[validation.neighbourhood_cleansed == 'Camden', 'price'].std()
validation.loc[validation.neighbourhood_cleansed == 'Camden', 'price'].hist(bins=20)

get_results(validation[validation.neighbourhood_cleansed == 'Islington'], X_feat_rfr)
validation_results = get_results(validation, X_feat_rfr)
validation_results
avg_weighted(validation_results, w2)
avg_weighted(validation_results[validation_results.N > 100], w2)

sns.scatterplot(validation_results.N, validation_results.test_rmse)


### Live data

live_model = pickle.load(open('models/Final_model_Hackney.pickle', 'rb'))
live_results = get_results(hackney, X_feat_rfr)
live_results

live_pred = live_model.predict(hackney[X_feat_rfr])
live_error = live_pred - hackney[y_var]

sum(live_error < -100) / hackney.shape[0]

# Check outliers
hackney.loc[hackney.price >= 1000, 'price'].sort_values()

# Try the prediction without the outliers
hackney_2 = hackney[hackney.price < 1000].copy()

live_results_2 = get_results(hackney_2, X_feat_rfr)
live_results_2

live_pred_2 = live_model.predict(hackney_2[X_feat_rfr])
live_error_2 = live_pred_2 - hackney_2[y_var]

ax = sns.distplot(live_error_2, label='error')
ax.set(xlabel='Live prediction error', ylabel='Frequency')


### Manchester data

manchester_raw = pd.read_csv('data/airbnb_manchester_cleaned.csv', index_col=0)
manchester = cleaner.clean_airbnb_data(manchester_raw)
manchester = cleaner.airbnb_impute(manchester, [x[:-8] for x in ab2.columns if x.endswith('_missing')])

man_feat = [x for x in X_feat_rfr if x in manchester.columns if x != 'calculated_host_listings_count']
rfr_man = RandomForestRegressor(n_estimators=1000, max_features='sqrt', n_jobs=2)
rfr_man.fit(ab2[man_feat], ab2['price'])

manchester_pred = rfr_man.predict(manchester[man_feat])

rmse_man = np.sqrt(mean_squared_error(manchester.price, manchester_pred))
r2_man = rfr_man.score(manchester[man_feat], manchester.price)

print(f"For the Manchester data, the RMSE is {rmse_man}, and the R2 is {r2_man}.")

ax = sns.distplot(manchester_pred - manchester.price)
ax.set(xlabel='Live prediction error', ylabel='Frequency')

np.mean(manchester_pred - manchester.price)
np.mean(manchester_pred / manchester.price)

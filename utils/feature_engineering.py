import pandas as pd
import numpy as np
from sklearn import preprocessing


def add_new_features(abt):
    abt = abt.copy()

    # Create additional features
    # host_listings_count
    abt['host_listings_count_diff'] = abt.calculated_host_listings_count - abt.host_listings_count

    # bedrooms
    abt.drop(columns='bedrooms_missing', inplace=True)
    bedroom_dummies = pd.get_dummies(abt.bedrooms.clip(0, 4))
    to_rename = [1 < float(x) < 2 for x in bedroom_dummies.columns]
    bedroom_dummies.columns = [d if not r else 'missing' for (d, r) in zip(bedroom_dummies.columns, to_rename)]
    bedroom_dummies.columns = [f'bedroom_{x}' for x in bedroom_dummies]
    abt = pd.concat([abt, bedroom_dummies.iloc[:, 1:]], axis=1)

    # accommodates
    acc_dummies = pd.get_dummies(abt.accommodates)
    acc_dummies.columns = [f'accommodates_{x}' for x in acc_dummies.columns]
    abt = pd.concat([abt, acc_dummies.iloc[:, 1:]], axis=1)

    # bathrooms
    abt.drop(columns='bathrooms_missing', inplace=True)
    bathroom_dummies = abt.bathrooms.clip(0, 3).copy()
    bathroom_dummies.loc[(bathroom_dummies > 1) & (bathroom_dummies < 1.5)] = -5
    bathroom_dummies = bathroom_dummies.astype(int)
    bathroom_dummies = pd.get_dummies(bathroom_dummies)
    bathroom_dummies.rename(columns={-5: 'missing'}, inplace=True)
    bathroom_dummies.columns = [f'bathroom_{str(x)}' for x in bathroom_dummies.columns]
    abt = pd.concat([abt, bathroom_dummies.iloc[:, 1:]], axis=1)

    # security_deposit
    abt['ln_security_deposit'] = np.log(abt.security_deposit)
    abt.loc[abt.ln_security_deposit == -np.inf, 'ln_security_deposit'] = -5

    # maximum_nights
    nnights_dummies = pd.get_dummies(pd.cut(abt.maximum_nights, [0, 20, 184, np.inf]))
    nnights_dummies.columns = [f'max_nights_{x}' for x in nnights_dummies.columns]
    abt = pd.concat([abt, nnights_dummies.iloc[:, 1:]], axis=1)

    # interactions
    abt['accommodates_X_entire_apartment'] = abt.accommodates * abt['Entire home/apt']
    abt['bedrooms_X_entire_apartment'] = abt.bedrooms * abt['Entire home/apt']
    abt['bathrooms_X_entire_apartment'] = abt.bathrooms * abt['Entire home/apt']
    abt['review_location_X_entire_apartment'] = abt.review_scores_location * abt['Entire home/apt']
    abt['review_cleanliness_X_entire_apartment'] = abt.review_scores_cleanliness * abt['Entire home/apt']
    abt['ln_deposit_X_entire_apartment'] = np.log(abt.security_deposit).clip(-5, 100) * abt['Entire home/apt']
    abt['bedrooms_X_bathrooms'] = abt.bathrooms * abt.bedrooms
    abt['accommodates_per_bedrooms'] = (abt.accommodates / abt.bedrooms).clip(0, 10)
    abt['accommodates_per_bathrooms'] = (abt.accommodates / abt.bathrooms).clip(0, 10)

    cols_to_drop = (['calculated_host_listings_count', 'review_scores_rating',
                     'bathrooms', 'bedrooms', 'Private room',
                     'accommodates', 'strict', 'maximum_nights']
                    + abt.neighbourhood_cleansed.unique().tolist())
    abt.drop(columns=cols_to_drop, inplace=True)

    return abt

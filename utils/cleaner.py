import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split


def clean_airbnb_data(ab, london=True):

    # Drop out-of-scope observations
    filter_1 = (ab.accommodates <= 6)
    filter_2 = (ab.property_type == 'Apartment')
    ab = ab[filter_1 & filter_2].copy()

    # Make numerical values numeric
    for n in ['price', 'security_deposit', 'weekly_price']:
        if ab[n].dtypes == 'object':
            ab[n] = pd.to_numeric(ab[n].str.replace(',', ''))

    # Drop invalid prices
    ab = ab[(ab.price < ab.weekly_price) | (ab.weekly_price.isna())]

    # Create dummies from factor variables
    ab['cancellation_policy'] = \
        ab['cancellation_policy'].replace({'super_strict_30': 'strict',
                                           'super_strict_60': 'strict'})
    for d in ['neighbourhood_cleansed', 'room_type', 'cancellation_policy']:
        ab = pd.concat([ab, pd.get_dummies(ab[d])], axis=1)

    # Create numeric variables from dates (days before 2017.04.01.)
    now = dt.date(2017, 4, 1)
    for d in ['host_since', 'first_review', 'last_review']:
        ab[d] = (pd.to_numeric((pd.to_datetime(now) \
            - pd.to_datetime(ab[d])).head()) / 3600 / 24 / 1000000000).astype(int)

    # Drop useless cols
    useless_cols = ['host_name', 'neighbourhood', 'city', 'state', 'market',
                    'smart_location', 'host_id', 'scrape_id', 'weekly_price',
                    'requires_license', 'host_total_listings_count']
    ab.drop(columns=useless_cols, inplace=True)

    # Create additional columns
    ab['calendar_never_updated'] = ab.calendar_updated == 'never'
    ab['calendar_updated'] = ab.calendar_updated.apply(clean_update)

    if london:
        # Adding data about London borough groups (dummy for each)
        superbrgs = {'inner_east': ['Haringey', 'Islington', 'Lambeth', 'Lewisham',
                        'Newham', 'Southwark', 'Tower Hamlets'],
                     'inner_west': ['Camden',
                                    'City of London',
                                    'Hammersmith and Fulham',
                                    'Kensington and Chelsea',
                                    'Wandsworth',
                                    'Westminster'],
                      'outer_w_nw': ['Barnet',
                                     'Brent',
                                     'Ealing',
                                     'Harrow',
                                     'Hillingdon',
                                     'Hounslow',
                                     'Richmond upon Thames'],
                      'outer_e_ne': ['Barking and Dagenham',
                                     'Bexley',
                                     'Enfield',
                                     'Greenwich',
                                     'Havering',
                                     'Redbridge',
                                     'Waltham Forest'],
                      'outer_south': ['Bromley',
                                      'Croydon',
                                      'Kingston upon Thames',
                                      'Merton',
                                      'Sutton']}

        for k, v in superbrgs.items():
            ab[k] = ab.neighbourhood_cleansed.isin(v)

    return ab


def split_airbnb_data(ab):

    np.random.seed(2020)

    # Hackney and non-Hackney split
    hackney = ab[ab.neighbourhood_cleansed == 'Hackney'].copy()
    ab = ab[ab.neighbourhood_cleansed != 'Hackney'].copy()
    ab, validation = train_test_split(ab, test_size=0.1, stratify=ab.neighbourhood_cleansed)

    return ab, validation, hackney


def clean_abt_parts(ab, validation, hackney, brgs):

    ab = ab.copy()
    validation = validation.copy()
    hackney = hackney.copy()

    # Drop variables with too many unknown values
    nas = ab.isna().sum(axis=0) / ab.shape[0]
    ab.drop(columns=nas[nas > 0.75].index.tolist(), inplace=True)
    validation.drop(columns=nas[nas > 0.75].index.tolist(), inplace=True)
    hackney.drop(columns=nas[nas > 0.75].index.tolist(), inplace=True)

    ab = airbnb_impute(ab)
    missing_cols = [x[:-8] for x in ab.columns if x.endswith('_missing')]
    validation = airbnb_impute(validation, missing_cols)
    hackney = airbnb_impute(hackney, missing_cols)

    def merge_brgs(df, brgs):
        return df.merge(brgs[['borough', 'inner_district']],
                      how='left',
                      left_on='neighbourhood_cleansed',
                      right_on='borough')

    ab = merge_brgs(ab, brgs)
    validation = merge_brgs(validation, brgs)
    hackney = merge_brgs(hackney, brgs)

    return ab, validation, hackney


def clean_update(s):
    if s.endswith('days ago'):
        return int(s.split(' ')[0])
    elif s.endswith('weeks ago'):
        return int(s.split(' ')[0]) * 7
    elif s.endswith('months ago'):
        return int(s.split(' ')[0]) * 30
    elif s == 'today':
        return 0
    elif s == 'yesterday':
        return 1
    elif s.endswith('week ago'):
        return 7
    elif s == 'never':
        return 2000
    else:
        return None


def clean_borough_data(brgs):

    # Name conventions
    brgs.borough.replace({'Barking': 'Barking and Dagenham',
                          'Hammersmith': 'Hammersmith and Fulham'},
                         inplace=True)
    brgs = brgs.append(pd.DataFrame({'borough': ['City of London'],
                                     'type': ['Inner']}))

    # Flag for inner district
    brgs['inner_district'] = brgs.type == 'Inner'

    return brgs


def airbnb_impute(df, missing_cols=None):
    # Impute missing values
    impute_zeros = (['security_deposit', 'cleaning_fee', 'reviews_per_month',
                     'host_is_superhost', 'host_listings_count',
                     'host_has_profile_pic', 'host_identity_verified']
                    + [x for x in df.columns if x.startswith('review_scores')])
    impute_min = ['review_scores_rating']
    impute_mean = ['bathrooms', 'bedrooms', 'beds', 'host_response_rate']

    nas = df.isna().sum(axis=0)

    if not missing_cols:
        missing_cols = df[nas[nas > 50].index].select_dtypes('number').columns
    for col in missing_cols:
        df[col+'_missing'] = df[col].isna()

    for col in impute_zeros:
        df[col] = df[col].fillna(0)
    for col in impute_min:
        df[col] = df[col].fillna(df[col].min())
    for col in impute_mean:
        df[col] = df[col].fillna(df[col].mean())

    return df

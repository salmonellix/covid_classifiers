from __future__ import print_function
import pandas as pd
import re
import sys
import math
import docopt
from math import sqrt
from time import strftime
from datetime import timedelta
import matplotlib.pyplot as plt
from dateutil.parser import parse
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
import os
import git
import numpy as np
from tqdm import tqdm
from datetime import datetime, date, time
# FORECASTING
#


DAYS_IN_FUTURE = 10 # Amount of days you want to forecast into future
PERC_SPLIT = 0.95   # Train / test split for forecasting model.

#
# DATA VISUALIZATION
#
FIG_SIZE = (14,10)


#Github cols
KEEP_COLS = ['country',
             'province',
             'confirmed',
             'deaths',
             'recovered',
             'date',
             'datetime',
             'file_date']

NUMERIC_COLS = ['confirmed',
                'deaths',
                'recovered']




def fix_country_names(tmp_df):
    '''
    Cleaning up after JHU's bullshit data management
    '''
    # Asian Countries
    tmp_df['country'] = np.where((tmp_df['country'] == 'Mainland China'), 'China', tmp_df['country'])
    tmp_df['country'] = np.where((tmp_df['country'] == 'Korea, South'), 'South Korea', tmp_df['country'])
    tmp_df['country'] = np.where((tmp_df['country'] == 'Republic of Korea'), 'South Korea', tmp_df['country'])
    tmp_df['country'] = np.where((tmp_df['country'] == 'Hong Kong SAR'), 'Hong Kong', tmp_df['country'])
    tmp_df['country'] = np.where((tmp_df['country'] == 'Taipei and environs'), 'Taiwan', tmp_df['country'])
    tmp_df['country'] = np.where((tmp_df['country'] == 'Taiwan*'), 'Taiwan', tmp_df['country'])
    tmp_df['country'] = np.where((tmp_df['country'] == 'Macao SAR'), 'Macau', tmp_df['country'])
    tmp_df['country'] = np.where((tmp_df['country'] == 'Iran (Islamic Republic of)'), 'Iran', tmp_df['country'])
    tmp_df['country'] = np.where((tmp_df['country'] == 'Viet Nam'), 'Vietnam', tmp_df['country'])

    # European Countries
    tmp_df['country'] = np.where((tmp_df['country'] == 'UK'), 'United Kingdom', tmp_df['country'])
    tmp_df['country'] = np.where((tmp_df['country'] == ' Azerbaijan'), 'Azerbaijan', tmp_df['country'])
    tmp_df['country'] = np.where((tmp_df['country'] == 'Bosnia and Herzegovina'), 'Bosnia', tmp_df['country'])
    tmp_df['country'] = np.where((tmp_df['country'] == 'Czech Republic'), 'Czechia', tmp_df['country'])
    tmp_df['country'] = np.where((tmp_df['country'] == 'Republic of Ireland'), 'Ireland', tmp_df['country'])
    tmp_df['country'] = np.where((tmp_df['country'] == 'North Ireland'), 'Ireland', tmp_df['country'])
    tmp_df['country'] = np.where((tmp_df['country'] == 'Republic of Moldova'), 'Moldova', tmp_df['country'])
    tmp_df['country'] = np.where((tmp_df['country'] == 'Russian Federation'), 'Russia', tmp_df['country'])

    # African Countries
    tmp_df['country'] = np.where((tmp_df['country'] == 'Congo (Brazzaville)'), 'Congo', tmp_df['country'])
    tmp_df['country'] = np.where((tmp_df['country'] == 'Congo (Kinshasa)'), 'Congo', tmp_df['country'])
    tmp_df['country'] = np.where((tmp_df['country'] == 'Republic of the Congo'), 'Congo', tmp_df['country'])
    tmp_df['country'] = np.where((tmp_df['country'] == 'Gambia, The'), 'Gambia', tmp_df['country'])
    tmp_df['country'] = np.where((tmp_df['country'] == 'The Gambia'), 'Gambia', tmp_df['country'])

    # Western Countries
    tmp_df['country'] = np.where((tmp_df['country'] == 'USA'), 'America', tmp_df['country'])
    tmp_df['country'] = np.where((tmp_df['country'] == 'US'), 'America', tmp_df['country'])
    tmp_df['country'] = np.where((tmp_df['country'] == 'Bahamas, The'), 'The Bahamas', tmp_df['country'])
    tmp_df['country'] = np.where((tmp_df['country'] == 'Bahamas'), 'The Bahamas', tmp_df['country'])
    tmp_df['country'] = np.where((tmp_df['country'] == 'st. Martin'), 'Saint Martin', tmp_df['country'])
    tmp_df['country'] = np.where((tmp_df['country'] == 'St. Martin'), 'Saint Martin', tmp_df['country'])

    # Others
    tmp_df['country'] = np.where((tmp_df['country'] == 'Cruise Ship'), 'Others', tmp_df['country'])
    tmp_df = tmp_df.drop(columns=["lat", "long", "province"])
    grouped_df = tmp_df.groupby(["country"])
    tmp_df = grouped_df.sum()
    # tmp_df = tmp_df.set_index("country")

    return tmp_df


def clean_data(df):
    tmp_df = df.copy()

    if 'Demised' in tmp_df.columns:
        tmp_df.rename(columns={'Demised': 'deaths'}, inplace=True)

    if 'Country/Region' in tmp_df.columns:
        tmp_df.rename(columns={'Country/Region': 'country'}, inplace=True)

    if 'Country_Region' in tmp_df.columns:
        tmp_df.rename(columns={'Country_Region': 'country'}, inplace=True)

    if 'Province/State' in tmp_df.columns:
        tmp_df.rename(columns={'Province/State': 'province'}, inplace=True)

    if 'Province_State' in tmp_df.columns:
        tmp_df.rename(columns={'Province_State': 'province'}, inplace=True)

    if 'Last Update' in tmp_df.columns:
        tmp_df.rename(columns={'Last Update': 'datetime'}, inplace=True)

    if 'Last_Update' in tmp_df.columns:
        tmp_df.rename(columns={'Last_Update': 'datetime'}, inplace=True)

    # Lower case all col names
    tmp_df.columns = map(str.lower, tmp_df.columns)

    #
    # for col in tmp_df[4:]:
    #     tmp_df[col] = tmp_df[col].fillna(0)
    #     tmp_df[col] = tmp_df[col].astype(int)

    return tmp_df

def get_data(cleaned_sheets):
    all_csv = []
    # Import all CSV's
    for tmp_df in cleaned_sheets:
            # try:
            #     tmp_df = pd.read_csv(os.path.join(DATA, f), index_col=None, header=0, parse_dates=['Last Update'])
            # except:
            #     # Temporary fix for JHU's bullshit data management
            #     tmp_df = pd.read_csv(os.path.join(DATA, f), index_col=None, header=0, parse_dates=['Last_Update'])

            tmp_df = clean_data(tmp_df)
            # tmp_df = tmp_df[KEEP_COLS]
            tmp_df['province'].fillna(tmp_df['country'], inplace=True)  # If no region given, fill it with country
            all_csv.append(tmp_df)

    df_raw = pd.concat(all_csv, axis=0, ignore_index=True, sort=True)  # concatenate all csv's into one df
    df_raw = fix_country_names(df_raw)  # Fix mispelled country names
    # df_raw = df_raw.sort_values(by=['datetime'])

    return df_raw



url_ded = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
url_conf = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
url_rec = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"




df_deaths = pd.read_csv(url_ded, error_bad_lines=False)
df_confirmed = pd.read_csv(url_conf, error_bad_lines=False)
df_recovered = pd.read_csv(url_rec, error_bad_lines=False)

df_deaths=clean_data(df_deaths)
df_confirmed=clean_data(df_confirmed)
df_recovered=clean_data(df_recovered)

df_deaths=fix_country_names(df_deaths)
df_confirmed=fix_country_names(df_confirmed)
df_recovered=fix_country_names(df_recovered)

# df_confirmed=df_confirmed-df_deaths-df_recovered
result = pd.concat([df_deaths, df_confirmed,df_recovered], axis=1, sort=False)
df_raw = df_confirmed


def forecast(tmp_df, t, v, days_in_future):
    train = t
    valid = v
    index_forecast = [x for x in range(0, len(valid) + days_in_future + 1)]

    # Fit model with training data
    model = auto_arima(train, trace=False, error_action='ignore', suppress_warnings=True)
    model_fit = model.fit(train)

    forecast, confint = model_fit.predict(n_periods=len(index_forecast), return_conf_int=True)
    RMSE = sqrt(mean_squared_error(valid, forecast[:len(valid)]))
    print('... RMSE:', RMSE)
    print('... forecasting {} days in the future'.format(days_in_future))

    # For plotting date ranges
    date_1 = 0
    date_2 = len(tmp_df["China"])
    # date_range = [d.strftime('%Y-%m-%d') for d in pd.date_range(date_1, date_2)]
    date_range = pd.date_range(date_1, date_2)

    forecast = pd.DataFrame(forecast, index=index_forecast, columns=['pred'])
    tmp_df = pd.concat([tmp_df, forecast], axis=1, sort=False)
    # tmp_df['date'] = pd.Series(date_range).astype(str)
    tmp_df[''] = None  # Dates get messed up, so need to use pandas plotting

    # # Save Model and file
    # print('... saving file:')
    tmp_df.to_csv("forecast_poland2.csv")
    # pl = pd.concat([train,forecast],axis=1)
    # plt.plot(pl)
    # plt.show()
    # plot_forecast(tmp_df, train, valid, index_forecast, forecast, confint)


def split_data(tmp_df, days_in_future,country):
    print('... train/test split:', PERC_SPLIT)
    tmp_df = tmp_df.T
    print(tmp_df)
    # tmp_df.to_csv("all2.csv")
    df_country = tmp_df[country]
    # Split data
    train = df_country[:int(PERC_SPLIT * (len(df_country)))]
    valid = df_country[int(PERC_SPLIT * (len(df_country))):]

    #
    forecast(tmp_df, train, valid, days_in_future)


pr = split_data(df_raw,10,"Poland")
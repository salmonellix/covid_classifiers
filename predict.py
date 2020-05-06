from __future__ import print_function
import re
import os
import sys
import math
import docopt
import pandas as pd
from math import sqrt
from time import strftime
from datetime import timedelta
import matplotlib.pyplot as plt
from dateutil.parser import parse
from pmdarima.arima import auto_arima
from datetime import datetime, date, time
from sklearn.metrics import mean_squared_error

# FORECASTING
#
DAYS_IN_FUTURE = 10 # Amount of days you want to forecast into future
PERC_SPLIT = 0.95   # Train / test split for forecasting model.

#
# DATA VISUALIZATION
#
FIG_SIZE = (14,10)


font = {'weight': 'bold',
        'size': 22}

plt.rc('font', **font)
plt.style.use('ggplot')
pd.options.display.max_rows = 999

args = docopt.docopt(__doc__)
out = args['--output_folder']
days_in_future = int(args['--num_days'])

# file paths
image_dir = os.path.join(out, 'reports', 'images')
trend_file = 'trend_{}.csv'.format(datetime.date(datetime.now()))
forecast_file = 'forecast_{}.csv'.format(datetime.date(datetime.now()))
data_dir = os.path.join(out, 'data', str(datetime.date(datetime.now())))

trend_df = pd.read_csv(os.path.join(data_dir, trend_file))

if not os.path.exists(image_dir):
    print('Creating reports folder...')
    os.system('mkdir -p ' + image_dir)


def plot_forecast(tmp_df, train, valid, index_forecast, forecast, confint):
    '''
    Plot the values of train and test, the predictions from ARIMA and the shadowing
    for the confidence interval.

    '''

    # For shadowing
    lower_series = pd.Series(confint[:, 0], index=index_forecast)
    upper_series = pd.Series(confint[:, 1], index=index_forecast)

    print('... saving graph')
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    plt.title('ARIMA - Prediction for cumalitive case counts {} days in the future'.format(days_in_future))
    plt.plot(train, label='Train', marker='o')
    plt.plot(valid, label='Test', marker='o')
    plt.plot(forecast, label='Forecast', marker='o')
    tmp_df.groupby('date')[['']].sum().plot(ax=ax)
    plt.fill_between(index_forecast,
                     upper_series,
                     lower_series,
                     color='k', alpha=.1)
    plt.ylabel('Infections')
    plt.xlabel('Date')
    fig.legend().set_visible(True)
    fig = ax.get_figure()
    fig.savefig(os.path.join(image_dir, 'cumulative_forecasts.png'))


def forecast(tmp_df, t, v, days_in_future):
    train = t.cumulative_cases
    valid = v.cumulative_cases
    index_forecast = [x for x in range(valid.index[0], valid.index[-1] + days_in_future + 1)]

    # Fit model with training data
    model = auto_arima(train, trace=False, error_action='ignore', suppress_warnings=True)
    model_fit = model.fit(train)

    forecast, confint = model_fit.predict(n_periods=len(index_forecast), return_conf_int=True)
    RMSE = sqrt(mean_squared_error(valid, forecast[:len(valid)]))
    print('... RMSE:', RMSE)
    print('... forecasting {} days in the future'.format(days_in_future))

    # For plotting date ranges
    date_1 = datetime.strptime(tmp_df.date.min(), "%Y-%m-%d")
    date_2 = datetime.strptime(tmp_df.date.max(), "%Y-%m-%d") + timedelta(days=days_in_future)
    date_range = [d.strftime('%Y-%m-%d') for d in pd.date_range(date_1, date_2)]
    date_range = pd.date_range(date_1, date_2)

    forecast = pd.DataFrame(forecast, index=index_forecast, columns=['pred'])
    tmp_df = pd.concat([tmp_df, forecast], axis=1, sort=False)
    tmp_df['date'] = pd.Series(date_range).astype(str)
    tmp_df[''] = None  # Dates get messed up, so need to use pandas plotting

    # Save Model and file
    print('... saving file:', forecast_file)
    tmp_df.to_csv(os.path.join(data_dir, forecast_file))

    plot_forecast(tmp_df, train, valid, index_forecast, forecast, confint)


def split_data(tmp_df):
    print('... train/test split:', PERC_SPLIT)
    # Split data
    train = tmp_df[:int(PERC_SPLIT * (len(tmp_df)))]
    valid = tmp_df[int(PERC_SPLIT * (len(tmp_df))):]

    forecast(tmp_df, train, valid, days_in_future)


if __name__ == '__main__':
    print('Training forecasting model...')
    split_data(trend_df)


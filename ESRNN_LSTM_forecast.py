import config
import sys
sys.path.append(config.ESRNN_PATH)

import datetime
import pandas as pd

from ESRNN import ESRNN


def forecast(train, df_network, product, validation=False):
    """
    Trains & predicts an ESRNN model for a single product

    Parameters
    ----------
    train: DataFrame
        DataFrame with `config.DATE_COLUMN`, `config.IMPUTED_QTY` and `config.MEDICINE_ITEM_ID` columns at minimum.
    df_network: DataFrame
        
    product: str
        Unique identifier of medicine item to be forecasted
    validation: bool, optional
        Determines whether this is a validation or test run (default is False)

    Returns
    -------
    DataFrame
        Forecasted values for the given product
    """
    
    if validation:
        train_period = config.TRAIN_END
    else:
        train_period = config.TEST_END

    # ESRNN-specific cleaning
    train = train.rename(columns={config.DATE_COLUMN: 'ds', config.IMPUTED_QTY: 'y', config.MEDICINE_ITEM_ID: 'unique_id'})
    train = train.reset_index(drop=True)  # ESRNN package requires a reset index
    train['x'] = 'IN'  # placeholder value; currently no categorical variable as input

    # split into x-y dataframes
    X_df = train[['unique_id', 'ds', 'x']]
    y_df = train[['unique_id', 'ds', 'y']]

    # create test DF
    X_test_df = pd.DataFrame(pd.date_range(train_period+datetime.timedelta(weeks=1), 
                                    train_period+datetime.timedelta(weeks=config.FORECAST_HORIZON), 
                                    freq='w-fri'), columns=['ds'])
    X_test_df['unique_id'] = product
    X_test_df['x'] = ‘IN’


    # Instantiate model
    model = ESRNN(max_epochs=60, freq_of_test=10, batch_size=1, learning_rate=1e-3,
                  per_series_lr_multip=0.7, lr_scheduler_step_size=20,
                  lr_decay=0.05, gradient_clipping_threshold=75,
                  rnn_weight_decay=0.0, level_variability_penalty=200,
                  testing_percentile=100, training_percentile=100,
                  ensemble=False, max_periods=30, seasonality=[],
                  input_size=35, output_size=config.FORECAST_HORIZON,
                  cell_type='LSTM', state_hsize=50,
                  dilations=[[1], [6]], add_nl_layer=False,
                  random_seed=1, device='cpu', frequency='w-fri')

    try:
        model.fit(X_df, y_df, verbose=False)
        fcst_future = model.predict(X_test_df).drop(columns='x')
    except:
        return

    if validation:
        fcst_future = fcst_future.merge(df_network.rename(columns={config.DATE_COLUMN: 'ds', 
                                                             config.IMPUTED_QTY: 'y', 
                                                             config.MEDICINE_ITEM_ID: 'unique_id'}), 
                                     on=['unique_id', 'ds'], how='left')
        
    fcst_future = fcst_future.rename(columns={'unique_id': config.MEDICINE_ITEM_ID, 'ds': config.DATE_COLUMN, 'y_hat': 'yhat'})
    return fcst_future








"""LSTM"""

from numpy import concatenate
from pandas import concat
from pandas import merge
from pandas import to_datetime
from pandas import DataFrame

from datetime import timedelta
from keras.layers import LSTM

from keras.models import Sequential
from keras.layers import concatenate

from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

import config
import tensorflow as tf


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def fit(train):
    scaler: ndarray of shape (n_features,)

    training_y = train[[config.IMPUTED_QTY]]
    values = training_y.values
    
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # frame as supervised learning
    # for lag = 1
    # reframed = series_to_supervised(scaled, 1, 1)
    # for lag = 5
    reframed = series_to_supervised(scaled, 5, 1)

    # split into train and test sets
    values = reframed.values
    n_train = len(values) - 5
    train = values[:n_train, :]
    test = values[n_train:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    # design network
    model = Sequential()
    model.add(LSTM(60, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    # fit network
    history = model.fit(train_X, train_y, epochs=75, batch_size=15, validation_data=(test_X, test_y), verbose=0, shuffle=False)


def predict(model, scaler, medicine_item, df_network, validation=True):
    """Predicts for a fitted LSTM Model

    Parameters
    ----------
    model: model object
        Fitted LSTM model.

    scaler: ndarray of shape (n_features,)
        Per feature relative scaling of the data.

    medicine_item: string
        Medicine_item (Medicine_item Aggregate) for which forecast is being generated.

    df_network: Pandas DataFrame
        
    validation: Boolean

    """

    # forecast
    if validation:
        TRAIN_END = config.TRAIN_END
    else:
        TRAIN_END = config.TEST_END
        
    # Forecast df formation - with Lag = 4
    FCST_X_START = TRAIN_END - timedelta(weeks=config.FORECAST_HORIZON + 3)
    fcst = df_network[
        (df_network[config.DATE_COLUMN] >= FCST_X_START) & (df_network[config.DATE_COLUMN] <= TRAIN_END)]
    fcst = fcst[[config.IMPUTED_QTY]]

    values = fcst.values
    # normalize features
    scaled = scaler.fit_transform(values)

    # frame as supervised learning
    # for lag = 1
    # reframed = series_to_supervised(scaled, 1, 1)
    # for lag = 4
    reframed = series_to_supervised(scaled, 4, 1)

    # split into input and outputs        
    values = reframed.values

    pred = values
    pred_X = pred[:, :-1]
    # reshape input to be 3D [samples, timesteps, features]
    pred_X = pred_X.reshape((pred_X.shape[0], 1, pred_X.shape[1]))

    # make a prediction
    yhat_pred = model.predict(pred_X)
    pred_X = pred_X.reshape((pred_X.shape[0], pred_X.shape[2]))
    
    # invert scaling for forecast
    inv_yhat_pred = concatenate((yhat_pred, pred_X[:, :]), axis=1)
    inv_yhat_pred = scaler.inverse_transform(inv_yhat_pred)
    inv_yhat_pred = inv_yhat_pred[:, 0]

    df_predict = DataFrame(inv_yhat_pred)
    df_predict.columns = ["yhat"]
    df_predict["id"] = range(0, df_predict.shape[0])
    # for lag = 4
    # df_predict["id"] = df_predict["id"] + 4
    
    # Create Future DS
    for j in range(0, config.FORECAST_HORIZON):
        date = TRAIN_END + timedelta(days=7 * (j + 1))
        if j == 0:
            future = DataFrame([date])
        else:
            future = future.append([date])
            future = DataFrame(future)
    future.columns = [config.DATE_COLUMN]
    future[config.DATE_COLUMN] = to_datetime(future[config.DATE_COLUMN])
    future["id"] = range(0, future.shape[0])
    future = merge(future, df_predict,on=["id"], how='left')
    future = future.drop(["id"], axis=1)
    fcst_future = future.copy()
    fcst_future[config.MEDICINE_ITEM_ID] = medicine_item

    if validation:
        # create dataframe prediction set
        # fcst_future = forecast[forecast[config.DATE_COLUMN] > config.TRAIN_END][[config.DATE_COLUMN, 'yhat']]
        # fcst_future[config.MEDICINE_ITEM_ID] = medicine_item
        # add actual network values
        temp_df = df_network[df_network[config.DATE_COLUMN] > TRAIN_END][[config.MEDICINE_ITEM_ID, config.DATE_COLUMN, config.QTY]]
        fcst_future = fcst_future.merge(temp_df,on=[config.DATE_COLUMN, config.MEDICINE_ITEM_ID], how='left')

    return fcst_future


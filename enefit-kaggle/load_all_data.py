import numpy as np
import pandas as pd
import os

#change to working directory
os.chdir("/Users/jacksonwalters/Documents/GitHub/enefit-kaggle/predict-energy-behavior-of-prosumers/")

#helper function to convert datetime strings to integers representing a time year-month-day hour-min-sec
from datetime import datetime
def datestr_to_int(datetime_str,date_format):
    if not pd.isna(datetime_str):
        return datetime.strptime(datetime_str, date_format).timestamp()
    else:
        return float('nan')

#function to load all data
def merged_df():

    print("loading train data...")
    #load the training data, dropping NaN's
    train = pd.read_csv("train.csv").dropna()
    train['datetime'] = train['datetime'].apply(lambda x: datestr_to_int(x,'%Y-%m-%d %H:%M:%S'))
    #shift data_block_id by +1 to line up with electricity_prices and gas_prices
    train['data_block_id'] += 1

    print("loading gas_prices...")
    #load gas_prices
    gas_prices = pd.read_csv("gas_prices.csv")
    #convert date strings to ints
    gas_prices['forecast_date'] = gas_prices['forecast_date'].apply(lambda x: datestr_to_int(x,'%Y-%m-%d'))
    gas_prices = gas_prices.drop(columns=['origin_date'])

    print("loading electricity_prices...")
    #load electricity_prices
    electricity_prices = pd.read_csv("electricity_prices.csv")
    #convert date strings to ints
    electricity_prices['forecast_date'] = electricity_prices['forecast_date'].apply(lambda x: datestr_to_int(x,'%Y-%m-%d %H:%M:%S'))
    electricity_prices = electricity_prices.drop(columns=['origin_date'])

    print("loading forecast_weather...")
    #load forecast_weather
    forecast_weather = pd.read_csv("forecast_weather.csv")
    #convert strings to ints
    forecast_weather['forecast_datetime'] = forecast_weather['forecast_datetime'].apply(lambda x: datestr_to_int(x,'%Y-%m-%d %H:%M:%S'))
    forecast_weather = forecast_weather.drop(columns=['origin_datetime'])
    forecast_weather = forecast_weather.rename(columns={'forecast_datetime':'forecast_date'})
    #shift times to line up with gas/electricity
    forecast_weather['forecast_date'] -= 10_800

    print("loading historical_weather...")
    #load forecast_weather
    historical_weather = pd.read_csv("historical_weather.csv")
    #convert strings to ints
    historical_weather['datetime'] = historical_weather['datetime'].apply(lambda x: datestr_to_int(x,'%Y-%m-%d %H:%M:%S'))
    #shift times to line up with gas/electricity

    #merge all the data

    print("merging train and gas_prices...")
    #merge gas prices and train.csv data
    #column names differ, so use left_on and right_on
    df = pd.merge(train, gas_prices, left_on=['data_block_id','datetime'], right_on=['data_block_id','forecast_date'], how='left')

    print("merging electricity_prices...")
    #merge train and gas_prices via left join on data_block_id
    #this leaves all rows of train, but matches
    df = df.merge(electricity_prices, on=['data_block_id','forecast_date'], how='left')

    print("merging forecast_weather...")
    #merge forecast_weather on forecast date
    df = df.merge(forecast_weather, on=['data_block_id','forecast_date'],how='left')

    print("merging historical_weather...")
    #merge historical_weather on datetime
    df = df.merge(historical_weather, left_on=['data_block_id','forecast_date'], right_on=['data_block_id','datetime'], how='left')

    #rename datetime to prediction datetime
    df = df.rename(columns={'datetime': 'prediction_datetime'})

    #drop NaN rows
    df = df.dropna()

    return df
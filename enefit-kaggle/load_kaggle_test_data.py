import pandas as pd
from datetime import datetime

#helper function to convert datetime strings to integers representing a time year-month-day hour-min-sec
#if datetime_str is nan, return float('nan')
def datestr_to_int(datetime_str,date_format):
    if not pd.isna(datetime_str):
        return datetime.strptime(datetime_str, date_format).timestamp()
    else:
        return float('nan')

#merge all of the data
def get_merged_df(test,electricity_prices,gas_prices,forecast_weather):
    #add a data_block_id column to merge on
    test['data_block_id'] = 0
    gas_prices['data_block_id'] = 0
    electricity_prices['data_block_id'] = 0
    forecast_weather['data_block_id'] = 0
    
    #convert test date_times to ints. times shifted by two days relative to data
    test['prediction_datetime'] = test['prediction_datetime'].apply(lambda x: datestr_to_int(str(x),'%Y-%m-%d %H:%M:%S'))
    #convert gas_price datetimes to ints. shift times by two days. drop origin_date.
    gas_prices = gas_prices.drop(columns=['origin_date'])
    gas_prices['forecast_date'] = gas_prices['forecast_date'].apply(lambda x: datestr_to_int(str(x),'%Y-%m-%d %H:%M:%S'))
    gas_prices['forecast_date'] += 172800
    #convert date strings to ints. shift times by two days. drop origin_date.
    electricity_prices = electricity_prices.drop(columns=['origin_date'])
    electricity_prices['forecast_date'] = electricity_prices['forecast_date'].apply(lambda x: datestr_to_int(str(x),'%Y-%m-%d %H:%M:%S'))
    electricity_prices['forecast_date'] += 172800
    #convert strings to ints. shift times by two days. drop origin_date. rename forecast_datetime to forecast_date
    forecast_weather = forecast_weather.drop(columns=['origin_datetime'])
    forecast_weather['forecast_datetime'] = forecast_weather['forecast_datetime'].apply(lambda x: datestr_to_int(str(x),'%Y-%m-%d %H:%M:%S'))
    forecast_weather = forecast_weather.rename(columns={'forecast_datetime':'forecast_date'})
    forecast_weather['forecast_date'] += 0.0
    
    #merge gas prices and train.csv data
    #column names differ, so use left_on and right_on
    df = pd.merge(test, gas_prices, on='data_block_id', how='left')
    #left join electricity_prices on origin_data and forecast_date
    df = df.merge(electricity_prices, on=['data_block_id','forecast_date'], how='left')
    #merge the forecast_weather data with the train+gas+electricity data
    df = df.merge(forecast_weather, on=['data_block_id','forecast_date'], how='left')
    #drop NaN rows
    df = df.dropna()
    
    return df
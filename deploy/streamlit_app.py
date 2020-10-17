import os
import random
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

# Base Functions

# Reset random seed
def reset_random_seeds(seed_value=1):
  os.environ['PYTHONHASHSEED']=str(seed_value)
  tf.random.set_seed(seed_value)
  np.random.seed(seed_value)
  random.seed(seed_value)

# Date-time Parsing
def parser(x):
  # return datetime.strptime(x, '%Y-%m-%d')
  return datetime.strptime(x, '%Y-%m-%d')

# Inverse scaling function
def inverse_scaling(series, scaler):
  temp = np.zeros((len(series), 5))
  temp[:, -1] = series.reshape(len(series))
  temp = scaler.inverse_transform(temp)
  inverted = temp[:, -1]
  return inverted

# Differencing
def difference(data, times=1):
  data_diff = data.diff()
  data_diff.dropna(inplace=True)
  if times == 0:
    return data
  elif times == 1:
    return data_diff
  else:
    for i in range(0,times-1):
      data_diff = data_diff.diff()
      data_diff.dropna(inplace=True)
    return data_diff

# Inverse difference
def inverse_difference(series, differenced, times_diff):
  for n in range(1,times_diff+1):
    inverted = list()
    seed = difference(series,times_diff-n)[-1]
    inverted.append(seed)
    for i in range(len(differenced)):
	    value = inverted[i] + differenced[i]
	    inverted.append(value)
    differenced = pd.Series(inverted[1:].copy())
  inverted_difference = differenced.copy()
  return inverted_difference

# Check available date
def check_available_date():
  start_in_range = True
  if start_date < start_limit_date:
    start_in_range = False
  out_of_range = False
  if start_date >= end_limit_date:
    print("Your request is out of range, we will predict some data for you!")
    out_of_range = True
  else:
    desired_dates = pd.date_range(start_date,end_date)
    for desired_date in desired_dates:
      if desired_date >= end_limit_date:
        print("Your request is out of range, we will predict some data for you!")
        out_of_range = True
        break
  return start_in_range, out_of_range

# Preprocess
def preprocess(supervised_values): #if out_of_range:
  # Scaling
  devide_border = int(0.8*(len(supervised_values)))
  supervised_values_head = supervised_values[:devide_border]
  supervised_values_tail = supervised_values[devide_border:]
  supervised_values_head = scaler.transform(supervised_values_head)
  supervised_values_tail = scaler.transform(supervised_values_tail)
  supervised_values_scaled = np.append(supervised_values_head, supervised_values_tail, axis=0)
  # Feature-label split
  feature, label = supervised_values_scaled[:, 0:-1], supervised_values_scaled[:, -1]
  # Feature reshape
  feature_reshaped = np.reshape(feature, (feature.shape[0], 2, 2, 1))
  # Make feature and label history
  feature_history = feature_reshaped
  label_history = label
  return feature_history, label_history

# Rolling predict
def rolling_predict(feature_history, label_history): #if out_of_range:
  predictions = np.array([])
  n = len(pd.date_range(end_limit_date,end_date))
  for i in range(n+1):
    # Train after first iteration
    if i > 0:
      reset_random_seeds()
      model.fit(feature_history, label_history, epochs=10, verbose=0)
    # Forecasting
    forecast = model.predict(np.array([feature_history[-1]]), verbose=0)
    # Append result to feature history for re-train
    temp = np.append(feature_history[-1].flatten()[1:], label_history[-1]).reshape((1,2,2,1))
    feature_history = np.append(feature_history, temp, axis=0)
    # Append result to label history for re-train
    label_history = np.append(label_history, forecast)
    # Collect predictions
    predictions = np.append(predictions, forecast)
  predictions = predictions[1:]
  return predictions

# Inverse predictions to original values
def inverse_predictions(predictions): #if out_of_range:
  # Inverse scaling
  predictions = inverse_scaling(predictions, scaler)
  # Inverse difference
  predictions = inverse_difference(dataset, predictions, 2)
  # Index = datetime
  predictions.index = pd.date_range(end_limit_date,end_date)
  return predictions

### LOAD FILES

# Title
title = "Thailand Covid-19 Confirmed Cases"

# Load dataset
data_path = 'thailand_covid_19_confirmed_cases.csv'
dataset = pd.read_csv(data_path, date_parser=parser, parse_dates=[0])
# Replace index with datetime
dataset = dataset.set_index('Date')
# Change from dataframe to series
dataset = pd.Series(dataset.values.flatten(), index=dataset.index)

# Load supervised values
supervised_values_path = 'thailand_covid_19_confirmed_cases_supervised_values.csv'
supervised_dataframe = pd.read_csv(supervised_values_path, date_parser=parser, parse_dates=[0])
# Replace index with datetime
supervised_dataframe = supervised_dataframe.set_index('Date')
# Get values
supervised_values = supervised_dataframe.values

# Load scaler
scaler_path = 'thailand-covid-19-confirmed-cases-scaler.pkl'
scaler = pickle.load(open(scaler_path, 'rb'))

# Load model
model_path = 'thailand-covid-19-confirmed-cases-model.h5'
model = tf.keras.models.load_model(model_path)

### SIDEBAR

st.sidebar.header("Sidebar")
# Input date
start_date = st.sidebar.date_input('start date', datetime(2020,1,22))
end_date = st.sidebar.date_input('start date', datetime(2020,7,27))

### PROCESS

# Convert input to datetime
start_date = datetime.combine(start_date, datetime.min.time())
end_date = datetime.combine(end_date, datetime.min.time())

# Limit (last date from dataset)
start_limit_date = datetime(2020,1,22)
end_limit_date = datetime(2020,7,28)

# Check available date
start_in_range, out_of_range = check_available_date()
# Start date warning
if start_in_range:
  print("Your start date request is in our range!")
else:
  print("We dont have it, please try another start date!")
# Start date in range, end date out of range
if start_in_range and out_of_range:
  feature_history, label_history = preprocess(supervised_values)
  predictions = rolling_predict(feature_history, label_history)
  predictions = inverse_predictions(predictions)
  if start_date >= end_limit_date:
    # Get data
    desired_data = predictions[start_date:]
    # Get df for plot
    pred_df = pd.DataFrame(predictions[start_date:], index=predictions[start_date:].index, columns=['Predictions'])
    desired_df = pd.concat([pred_df])
  else:
    # Get data
    desired_data = dataset[start_date:].append(predictions)
    # Bridge making
    bridge_index = [dataset.index[-1],predictions.index[0]]
    bridge = pd.Series([dataset[-1],predictions[0]], index=bridge_index)
    # Get df for plot
    exist_df = pd.DataFrame(dataset[start_date:], index=dataset[start_date:].index, columns=['Existing data'])
    bridge_df = pd.DataFrame(bridge, index=bridge.index, columns=['Bridge'])
    pred_df = pd.DataFrame(predictions, index=predictions.index, columns=['Predictions'])
    desired_df = pd.concat([exist_df, bridge_df, pred_df])
elif start_in_range:
  # Get data
  desired_data = dataset[start_date:end_date]
  # Get df for plot
  exist_df = pd.DataFrame(dataset[start_date:end_date], index=dataset[start_date:end_date].index, columns=['Existing data'])
  desired_df = pd.concat([exist_df])
  
 st.line_chart(desired_df)

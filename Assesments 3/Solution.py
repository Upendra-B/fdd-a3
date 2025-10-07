# ==========================================================
# Investigation B: Seasonal Behaviour Changes in Bats (Improved Visuals)
# ==========================================================

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats import proportion
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------
# STEP 1: Load and Prepare Data
# ----------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../Datasets")

df1 = pd.read_csv(os.path.join(DATA_DIR, "dataset1.csv"))
df2 = pd.read_csv(os.path.join(DATA_DIR, "dataset2.csv"))

# Convert columns to proper datatypes
for col in ['start_time', 'rat_period_start', 'rat_period_end', 'sunset_time']:
    df1[col] = pd.to_datetime(df1[col], errors='coerce')
for col in ['time']:
    df2[col] = pd.to_datetime(df2[col], errors='coerce')

df1[['bat_landing_to_food','hours_after_sunset','seconds_after_rat_arrival']] = df1[['bat_landing_to_food','hours_after_sunset','seconds_after_rat_arrival']].astype(float)
df2[['hours_after_sunset','bat_landing_number','food_availability','rat_minutes','rat_arrival_number']] = df2[['hours_after_sunset','bat_landing_number','food_availability','rat_minutes','rat_arrival_number']].astype(float)

df1.dropna(inplace=True)
df2.dropna(inplace=True)

# Replace numeric season with labels
df1['season'] = df1['season'].replace({0:'Winter', 1:'Spring'})

# Map months (1–6) to names for legend clarity
month_map = {0:'Dec', 1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun'}
df2['month'] = df2['month'].replace(month_map)

# ----------------------------------------------------------
# STEP 2: Descriptive Statistics
# ----------------------------------------------------------

print("==== Descriptive Statistics by Season ====")
print(df1.groupby('season')['risk'].mean())
print(df1.groupby('season')['bat_landing_to_food'].mean())
print(df2.groupby('month')[['rat_arrival_number','bat_landing_number']].mean())
print("==========================================\n")

# ----------------------------------------------------------
# STEP 3: Visual Analysis
# ----------------------------------------------------------

sns.set_style("whitegrid")

# Risk-taking by season
plt.figure(figsize=(7,5))
sns.barplot(x='season', y='risk', data=df1, palette=['#1f77b4', '#ff7f0e'])
plt.title("Average Risk-taking Behaviour by Season", fontsize=13)
plt.ylabel("Proportion of Risk-taking")
plt.xlabel("Season")
plt.show()

# Delay before food
plt.figure(figsize=(7,5))
sns.boxplot(x='season', y='bat_landing_to_food', data=df1, palette=['#66c2a5', '#fc8d62'])
plt.title("Bat Delay Before Approaching Food by Season", fontsize=13)
plt.ylabel("Delay to Food (seconds)")
plt.xlabel("Season")
plt.show()

# Rat arrivals by month
plt.figure(figsize=(7,5))
sns.barplot(x='month', y='rat_arrival_number', data=df2, palette='coolwarm')
plt.title("Average Rat Arrivals per Month", fontsize=13)
plt.ylabel("Mean Rat Arrivals (per 30 mins)")
plt.xlabel("Month")
plt.show()

# Bat landings by month
plt.figure(figsize=(7,5))
sns.barplot(x='month', y='bat_landing_number', data=df2, palette='viridis')
plt.title("Average Bat Landings per Month", fontsize=13)
plt.ylabel("Mean Bat Landings (per 30 mins)")
plt.xlabel("Month")
plt.show()


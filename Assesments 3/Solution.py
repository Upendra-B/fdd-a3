# ==========================================================
# Investigation B: Seasonal Behaviour Changes in Bats
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
# STEP 1 – Load & Clean Data
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../Datasets")

df1 = pd.read_csv(os.path.join(DATA_DIR, "dataset1.csv"))
df2 = pd.read_csv(os.path.join(DATA_DIR, "dataset2.csv"))

# Ensure correct types
time_cols_df1 = ['start_time','rat_period_start','rat_period_end','sunset_time']
for c in time_cols_df1: df1[c] = pd.to_datetime(df1[c], errors='coerce')
time_cols_df2 = ['time']
for c in time_cols_df2: df2[c] = pd.to_datetime(df2[c], errors='coerce')

num_df1 = ['bat_landing_to_food','hours_after_sunset','seconds_after_rat_arrival']
num_df2 = ['hours_after_sunset','bat_landing_number','food_availability','rat_minutes','rat_arrival_number']
df1[num_df1] = df1[num_df1].astype(float)
df2[num_df2] = df2[num_df2].astype(float)

df1.dropna(inplace=True)
df2.dropna(inplace=True)

# ----------------------------------------------------------
# STEP 2 – Descriptive Statistics
# ----------------------------------------------------------
print("\n=== Descriptive Statistics by Season ===")
print(df1.groupby('season')['risk'].mean())
print(df1.groupby('season')['bat_landing_to_food'].mean())
print(df2.groupby('month')[['rat_arrival_number','bat_landing_number']].mean())

# ----------------------------------------------------------
# STEP 3 – Visual Analysis (clear seasonal colour scheme)
# ----------------------------------------------------------
sns.set_style("whitegrid")

# 1️⃣ Risk-taking by season
plt.figure(figsize=(7,5))
sns.barplot(x='season', y='risk', data=df1, palette={'winter':'#1f77b4','spring':'#ff7f0e'})
plt.title("Average Risk-taking Behaviour by Season")
plt.xlabel("Season"); plt.ylabel("Proportion of Risk-taking")
plt.show()
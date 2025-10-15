# ============================================================
# HIT140 – Foundations of Data Science
# Assessment 3 | Linear Regression – Investigation A & B
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from statsmodels.stats import proportion
import statsmodels.api as sm

# ------------------------------------------------------------
# STEP 1: Load Datasets
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../Datasets")
df1_path = os.path.join(DATA_DIR, "dataset1.csv")
df2_path = os.path.join(DATA_DIR, "dataset2.csv")

df1 = pd.read_csv(df1_path)
df2 = pd.read_csv(df2_path)

# ------------------------------------------------------------
# STEP 2: Data Cleaning and Preparation
# ------------------------------------------------------------
# Convert timestamps
for col in ['start_time', 'rat_period_start', 'rat_period_end', 'sunset_time']:
    df1[col] = pd.to_datetime(df1[col], errors='coerce', dayfirst=True)
for col in ['time']:
    df2[col] = pd.to_datetime(df2[col], errors='coerce', dayfirst=True)

# Convert numerics
df1[['bat_landing_to_food','hours_after_sunset','seconds_after_rat_arrival']] = \
    df1[['bat_landing_to_food','hours_after_sunset','seconds_after_rat_arrival']].astype(float)
df2[['hours_after_sunset','bat_landing_number','food_availability','rat_minutes','rat_arrival_number']] = \
    df2[['hours_after_sunset','bat_landing_number','food_availability','rat_minutes','rat_arrival_number']].astype(float)

df1.dropna(inplace=True)
df2.dropna(inplace=True)

# Replace numeric season labels if present
df1['season'] = df1['season'].replace({0: 'Winter', 1: 'Spring'})

# Month mapping
month_map = {0:'Dec',1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun'}
df2['month_name'] = df2['month'].replace(month_map)

# Assign seasons for df2
def assign_season(month):
    if month in [0,1,2,3]:
        return 'Winter'
    elif month in [4,5,6]:
        return 'Spring'
    else:
        return 'Unknown'
df2['season'] = df2['month'].apply(assign_season)

sns.set_style("whitegrid")

# ============================================================
# INVESTIGATION A – Bat Behaviour and Rat Influence
# ============================================================
print("\n--- Investigation A: Behavioural Analysis ---")

X_A = df1[['risk', 'reward', 'seconds_after_rat_arrival']]
y_A = df1['bat_landing_to_food']
X_A_const = sm.add_constant(X_A)
modelA = sm.OLS(y_A, X_A_const).fit()
print(modelA.summary())

# --- Train/Test split for evaluation
X_train, X_test, y_train, y_test = train_test_split(X_A, y_A, test_size=0.3, random_state=42)
lr = LinearRegression().fit(X_train, y_train)
y_pred = lr.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
nrmse = rmse / (y_test.max() - y_test.min())
r2 = r2_score(y_test, y_pred)
print("\n--- Model Evaluation Metrics ---")
print(f"MAE: {mae:.3f} | MSE: {mse:.3f} | RMSE: {rmse:.3f} | NRMSE: {nrmse:.3f} | R²: {r2:.3f}")

# --- Multicollinearity (VIF)
vif = pd.DataFrame({
    "Variable": X_A.columns,
    "VIF": [variance_inflation_factor(X_A.values, i) for i in range(X_A.shape[1])]
})
print("\n--- Variance Inflation Factor (VIF) ---")
print(vif)

# --- Feature importance
importance = pd.DataFrame({
    'Variable': X_A_const.columns,
    'Coefficient': modelA.params
}).sort_values(by='Coefficient', ascending=False)
print("\n--- Feature Importance ---")
print(importance)

# --- Assumption checks (Residuals)
fitted_values = modelA.predict(X_A_const)
residuals = y_A - fitted_values

plt.figure(figsize=(7,5))
sns.scatterplot(x=fitted_values, y=residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--', label='Zero Residual Line')
plt.title("Residuals vs Fitted Values (Linearity & Homoscedasticity)")
plt.xlabel("Predicted Feeding Delay (seconds)")
plt.ylabel("Residuals (Actual - Predicted)")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
sns.histplot(residuals, kde=True, color='#ff7f0e')
plt.title("Residual Distribution (Checking Normality)")
plt.xlabel("Residual Value (seconds)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ============================================================
# INVESTIGATION B – Seasonal Comparison
# ============================================================
print("\n--- Investigation B: Seasonal Comparison ---")

# --- Descriptive statistics
print("==== Descriptive Statistics by Season ====")
print(df1.groupby('season')['risk'].mean())
print(df1.groupby('season')['bat_landing_to_food'].mean())
print(df2.groupby('month_name')[['rat_arrival_number', 'bat_landing_number']].mean())
print("==========================================\n")

# --- Seasonal visuals
plt.figure(figsize=(7,5))
sns.barplot(x='season', y='risk', hue='season', data=df1,
            palette={'Winter':'#1f77b4','Spring':'#ff7f0e'}, legend=False)
plt.title("Average Risk-taking Behaviour by Season")
plt.xlabel("Season (Winter = fewer rats, Spring = more rats)")
plt.ylabel("Mean Risk-taking (proportion)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
sns.boxplot(x='season', y='bat_landing_to_food', hue='season', data=df1,
            palette={'Winter':'#66c2a5','Spring':'#fc8d62'}, legend=False)
plt.title("Bat Delay Before Approaching Food by Season")
plt.xlabel("Season")
plt.ylabel("Delay Between Landing and Feeding (seconds)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(x='month_name', y='rat_arrival_number', hue='month_name', data=df2,
            palette='coolwarm', legend=False)
plt.title("Average Rat Arrivals per Month")
plt.xlabel("Month")
plt.ylabel("Mean Rat Arrivals (per 30 mins)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(x='month_name', y='bat_landing_number', hue='month_name', data=df2,
            palette='viridis', legend=False)
plt.title("Average Bat Landings per Month")
plt.xlabel("Month")
plt.ylabel("Mean Bat Landings (per 30 mins)")
plt.tight_layout()
plt.show()

# --- Inferential tests
delay_winter = df1[df1['season']=='Winter']['bat_landing_to_food']
delay_spring = df1[df1['season']=='Spring']['bat_landing_to_food']
t_stat, p_val = stats.ttest_ind(delay_winter, delay_spring, equal_var=False)
print("T-test: Delay to food (Winter vs Spring)")
print(f"t-statistic: {t_stat:.3f} | p-value: {p_val:.5f}\n")

winter_risk = df1[df1['season']=='Winter']['risk']
spring_risk = df1[df1['season']=='Spring']['risk']
count_winter, n_winter = winter_risk.sum(), len(winter_risk)
count_spring, n_spring = spring_risk.sum(), len(spring_risk)
ci_winter = proportion.proportion_confint(count_winter, n_winter, alpha=0.05, method='normal')
ci_spring = proportion.proportion_confint(count_spring, n_spring, alpha=0.05, method='normal')
print("95% Confidence Interval for Risk-taking:")
print("Winter:", ci_winter)
print("Spring:", ci_spring, "\n")

# --- Seasonal regressions (Winter vs Spring)
plt.figure(figsize=(8,6))
colors = {'Winter':'#1f77b4', 'Spring':'#ff7f0e'}

for season, color in colors.items():
    if season == 'Winter':
        df_season = df2[df2['month_name'].isin(['Jan','Feb','Mar'])]
    else:
        df_season = df2[df2['month_name'].isin(['Apr','May','Jun'])]
    X = df_season[['rat_minutes']]
    y = df_season['bat_landing_number']
    model = LinearRegression().fit(X, y)
    plt.plot(X, model.predict(X), color=color, label=f'{season} Regression Line')
    plt.scatter(X, y, color=color, alpha=0.5, label=f'{season} Observed Data')

plt.title("Linear Regression: Rat Presence vs Bat Landings by Season")
plt.xlabel("Rat Minutes (Total Rat Presence per 30-min Period)")
plt.ylabel("Bat Landings (Count per 30-min Period)")
plt.legend(title="Season\nWinter = Cold months, Spring = Warm months")
plt.tight_layout()
plt.show()

# --- Correlation per month
print("Correlation Between Rat Minutes and Bat Landings (per Month):")
for m in df2['month_name'].unique():
    subset = df2[df2['month_name'] == m]
    if len(subset) > 2:
        # Only calculate correlation if both columns have variation
        if subset['rat_minutes'].nunique() > 1 and subset['bat_landing_number'].nunique() > 1:
            corr, p = stats.pearsonr(subset['rat_minutes'], subset['bat_landing_number'])
            print(f"{m}: correlation = {corr:.3f}, p-value = {p:.5f}")
        else:
            print(f"{m}: correlation not defined (one variable is constant)")

# --- Regression performance by season
winter_df = df2[df2['season']=='Winter']
spring_df = df2[df2['season']=='Spring']
X_winter = winter_df[['hours_after_sunset','rat_minutes','food_availability']]
y_winter = winter_df['bat_landing_number']
X_spring = spring_df[['hours_after_sunset','rat_minutes','food_availability']]
y_spring = spring_df['bat_landing_number']
lr_winter = LinearRegression().fit(X_winter, y_winter)
lr_spring = LinearRegression().fit(X_spring, y_spring)
y_pred_winter = lr_winter.predict(X_winter)
y_pred_spring = lr_spring.predict(X_spring)
r2_winter = r2_score(y_winter, y_pred_winter)
r2_spring = r2_score(y_spring, y_pred_spring)
rmse_winter = np.sqrt(mean_squared_error(y_winter, y_pred_winter))
rmse_spring = np.sqrt(mean_squared_error(y_spring, y_pred_spring))
print(f"\nR² Winter: {r2_winter:.3f} | RMSE Winter: {rmse_winter:.3f}")
print(f"R² Spring: {r2_spring:.3f} | RMSE Spring: {rmse_spring:.3f}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n=== SUMMARY ===")
print("Winter Coefficients:", lr_winter.coef_)
print("Spring Coefficients:", lr_spring.coef_)
print("\nInterpretation:")
print("- Winter months (fewer rats, food scarcity) show more predictable bat landings.")
print("- Spring months (more rats, abundant food) show increased variability and interaction effects.")
print("\nAnalysis completed successfully.")

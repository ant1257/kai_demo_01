# Libraries
import h3 #current version is 3.7.6
import glob
import os
import sys
import pandas as pd
import joblib
import lightgbm as lgb
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
from math import ceil, floor
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set whole project visibility
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Custom functions/settings
from constants import DATA_DIR, TEMP_DIR, MODEL_DIR, RESOLUTION, part_of_day_labels, day_labels, check_time_list, features_col, target_col
from func.part_of_day import part_of_day
from func.season_of_year import season_of_year
from func.collect_data import collect_data


## Settings
# clear temp folder
print("Cleaning temp folder...")
parquet_files = glob.glob(os.path.join(TEMP_DIR, "*.parquet"))
for file_path in parquet_files:
    os.remove(file_path)
print(f"[OK] Removed {len(parquet_files)} .parquet files from {TEMP_DIR}")

id_date = str(datetime.today().date())
# id_date = '2025-03-01'
print("Train data id date: {}".format(id_date))
cutoff_start_date = datetime.strptime(id_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).date() - timedelta(days = 1)
cutoff_end_date = cutoff_start_date - relativedelta(months = 24)
data_selection_start_date = datetime.strptime(str(cutoff_start_date), "%Y-%m-%d").replace(tzinfo=timezone.utc).date()
data_selection_end_date = datetime.strptime(str(cutoff_end_date), "%Y-%m-%d").replace(tzinfo=timezone.utc).date()

print("List of dates for train data selection:")
start_date = cutoff_start_date - relativedelta(months = 3)
date_list = [
    (start_date + timedelta(days=i)).isoformat()
    for i in range((cutoff_start_date - start_date).days + 1)
]
for date in date_list:
    print(date)

start_date = data_selection_end_date
end_date = data_selection_start_date
months = []
current_date = start_date.replace(day=1)
while current_date <= end_date:
    months.append(current_date.strftime("%Y-%m"))
    current_date += relativedelta(months=1)


## Data collection
print("Reading necessary json files...")
df_original_all = pd.DataFrame()
for month in months:
    file_path = os.path.join(DATA_DIR, f"data-{month}-01.json")
    if os.path.exists(file_path):
        print("Reading {}...".format(file_path))
        df = pd.read_json(file_path)
        df_original_all = pd.concat([df_original_all, df], ignore_index=True)
    else:
        print(f"⚠️ File not found: {file_path}")
        
# Read all data
print("Feature engineering (1)...")
df_original_all["start_date_full"] = pd.to_datetime(df_original_all["SpecifiedStartDate"], format="ISO8601", utc=True)
df_original_all['start_date_year'] = df_original_all['start_date_full'].dt.year
df_original_all['start_date_month'] = df_original_all['start_date_full'].dt.month
df_original_all['start_date_season'] = df_original_all['start_date_month'].apply(season_of_year)
df_original_all['start_date_dow'] = df_original_all['start_date_full'].dt.dayofweek
df_original_all['is_weekend'] = df_original_all["start_date_dow"].isin([5, 6]).astype(int)
df_original_all['is_sunday'] = df_original_all["start_date_dow"].isin([6]).astype(int)
df_original_all['start_date'] = df_original_all['start_date_full'].dt.date
df_original_all['start_time'] = df_original_all['start_date_full'].dt.time
df_original_all['start_time_part_of_day'] = df_original_all['start_date_full'].apply(part_of_day)
df_original_all['h3_cell'] = df_original_all.apply(lambda row: h3.geo_to_h3(row['LatitudeStart'], row['LongitudeStart'], RESOLUTION), axis = 1)

df_original_sel = df_original_all[df_original_all['start_date'] <= cutoff_start_date].sort_values(by = ['SpecifiedStartDate']).reset_index(drop = True)
del df_original_all
print("Number of rows: {}".format(df_original_sel.shape[0]))


start = datetime.now()
print("Collecting aggregated data to be used for model training...")
print("Start time: {}".format(start))
print("NOTE1: date_list is the list of dates for the last 3 months.")
print("NOTE2: this may take an hour or so.")
for check_date_str in date_list:
    for check_time_str in check_time_list:
        collect_data(df_original_sel, check_date_str, check_time_str, TEMP_DIR, data_type = 'train')
end = datetime.now()
print(f"[OK] Data collected in {end - start}.")
del df_original_sel


## Train model and save
files = list()
print("Checking aggregated data for:")
for check_date_str in date_list:
    print(check_date_str)
    tmp_files = [f for f in os.listdir(TEMP_DIR) if f.startswith('train_{}_'.format(check_date_str.replace('-', '_')))]
    files.extend(tmp_files)
    del tmp_files
    
start_2 = datetime.now()
df_all = pd.DataFrame()
print("Reading files...")
print("NOTE: it may take some time")
for file in files:
    file_path = os.path.join(TEMP_DIR, file)
    if os.path.exists(file_path):
        df_temp = pd.read_parquet(os.path.join(TEMP_DIR, file))
        df_all = pd.concat([df_all, df_temp], ignore_index = True)
        del df_temp
    else:
        print(f"⚠️ File not found: {file_path}")
end_2 = datetime.now()
print(f"[OK] Data read in {end - start}.")
print("Number of rows: {}".format(df_all.shape[0]))

print("Feature engineering (2)...")
df_data_02 = df_all.drop(columns = ['is_sunday'])
df_data_02['day_of_week'] = df_data_02["prediction_date_time_start"].dt.dayofweek
df_data_02['is_weekend'] = df_data_02["day_of_week"].isin([5, 6]).astype(bool)
df_data_02['h3_cell_1_month_popularity'] = df_data_02['h3_cell_1_month_popularity'].fillna(0)
df_data_02['trip_count'] = df_data_02['trip_count'].fillna(0)
df_data_02['prev_1_hour_cnt'] = df_data_02['prev_1_hour_cnt'].astype(int)
df_data_02['prev_2_hour_cnt'] = df_data_02['prev_2_hour_cnt'].astype(int)
df_data_02['prev_3_hour_cnt'] = df_data_02['prev_3_hour_cnt'].astype(int)
df_data_02['part_of_day'] = df_data_02['part_of_day'].astype(int)
df_data_02['trip_count_1_year_back'] = df_data_02['trip_count_1_year_back'].astype(int)
df_data_02['trip_count'] = df_data_02['trip_count'].astype(int)

part_of_day_dummies = pd.get_dummies(df_data_02['part_of_day']).rename(columns=part_of_day_labels)
df_data_02 = pd.concat([df_data_02, part_of_day_dummies], axis=1)
for value in part_of_day_labels.values():
    if value not in df_data_02:
        df_data_02[value] = False

day_dummies = pd.get_dummies(df_data_02['day_of_week']).rename(columns=day_labels)
df_data_02 = pd.concat([df_data_02, day_dummies], axis=1)
for value in day_labels.values():
    if value not in df_data_02:
        df_data_02[value] = False

df_data_02 = df_data_02.drop(columns = ['part_of_day', 'day_of_week', 'is_early_morning', 'is_saturday'])

# Train a model
print("Starting model train...")
model_output_folder = os.path.join(MODEL_DIR, "model_train_{}_id".format(id_date.replace("-", "_")))
if not os.path.exists(model_output_folder):
    os.makedirs(model_output_folder)

X = df_data_02[features_col]
y = df_data_02[target_col]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lgb_model = lgb.LGBMRegressor(
    random_state=42,
    num_leaves = 70,
    learning_rate = 0.05,
    n_estimators = 500,
    subsample = 0.8,
    colsample_bytree = 0.8
)
lgb_model.fit(X_train, y_train)

# Save the model as a file
model_pkl_path = os.path.join(model_output_folder, "lgb_model_{}.pkl".format(id_date.replace("-", "_")))
print("Saving model to {}".format(model_pkl_path))
joblib.dump(lgb_model, model_pkl_path)
print("Saving model to main folder {} (the model to be used for predictions)".format(MODEL_DIR))
joblib.dump(lgb_model, os.path.join(MODEL_DIR, 'lgb_model.pkl'))

# check quality (test)
print("Quality metrics (test data):")
y_pred_lgb = lgb_model.predict(X_test)
lgb_mae = mean_absolute_error(y_test, y_pred_lgb)
print(f"MAE: {lgb_mae:.3f}")

lgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
print(f"RMSE: {lgb_rmse:.3f}")

lgb_r2 = r2_score(y_test, y_pred_lgb)
print(f"R²: {lgb_r2:.3f}")

metrics = {
    "id_date": str(id_date),
    "mae": float(lgb_mae),
    "rmse": float(lgb_rmse),
    "r2": float(lgb_r2),
    "n_test_samples": int(len(y_test)),
    "model_file": model_pkl_path
}
print("Saving quality metrics to {}".format(model_pkl_path))
metrics_path = os.path.join(model_output_folder, "metrics_{}.json".format(id_date.replace("-", "_")))
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)


print("Saving Actual vs Predicted figure to {}".format(model_pkl_path))
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lgb, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual ride_check_cnt")
plt.ylabel("Predicted ride_check_cnt")
plt.title("LightGBM Regressor: Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(model_output_folder, "lgb_actual_vs_predicted_{}.png".format(id_date.replace("-", "_"))), dpi=300, transparent=False, facecolor='white')
plt.close()



importance = lgb_model.feature_importances_
features = X_train.columns
feature_importance_df = pd.DataFrame({
    'feature': features,
    'importance': importance
}).sort_values(by='importance', ascending=False)
feature_importance_df['importance_pct'] = 100 * feature_importance_df['importance'] / feature_importance_df['importance'].sum()

print("Saving Feature Importance figure to {}".format(model_pkl_path))
plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance_df.head(20), x='importance_pct', y='feature', palette='viridis')
plt.title("Top 20 Feature Importances (LGBMRegressor)")
plt.xlabel("Importance (%)")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(os.path.join(model_output_folder, "lgb_feature_importance_{}.png".format(id_date.replace("-", "_"))), dpi=300, transparent=False, facecolor='white')
plt.close()
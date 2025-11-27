# Libraries
import h3 #current version is 3.7.6
import joblib
import os
import sys
import pandas as pd
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
from math import ceil, floor
from pathlib import Path

# Set whole project visibility
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Custom functions/settings
from constants import DATA_DIR, TEMP_DIR, MODEL_DIR, RESULT_DIR, RESOLUTION, part_of_day_labels, day_labels, check_time_list, features_col, target_col
from func.part_of_day import part_of_day
from func.season_of_year import season_of_year
from func.collect_data import collect_data

## Settings
# for test
full_date_time = datetime(2025, 5, 1, 10, 12, 0, tzinfo = timezone.utc)
# full_date_time = datetime.today()
id_date = str(full_date_time.date())
id_time = full_date_time.strftime("%H:%M:%S")
floored_minute = (full_date_time.time().minute // 10) * 10
id_time_floored = full_date_time.time().replace(minute=floored_minute, second=0, microsecond=0)
cutoff_start_date = datetime.strptime(id_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).date()
cutoff_end_date = cutoff_start_date - relativedelta(months = 24)
print("Score data id date and time: {} {}".format(id_date, id_time_floored))

data_selection_start_date = datetime.strptime(str(cutoff_start_date), "%Y-%m-%d").replace(tzinfo=timezone.utc).date()
data_selection_end_date = datetime.strptime(str(cutoff_end_date), "%Y-%m-%d").replace(tzinfo=timezone.utc).date()

start_date = cutoff_start_date - relativedelta(months = 3)
date_list = [
    (start_date + timedelta(days=i)).isoformat()
    for i in range((cutoff_start_date - start_date).days + 1)
]

start_date = data_selection_end_date
end_date = data_selection_start_date
months = []
current_date = start_date.replace(day=1)
while current_date <= end_date:
    months.append(current_date.strftime("%Y-%m"))
    current_date += relativedelta(months=1)
    
    
## Data collection
df_original_all = pd.DataFrame()
for month in months:
    file_path = os.path.join(DATA_DIR, f"data-{month}-01.json")
    if os.path.exists(file_path):
        df = pd.read_json(file_path)
        df_original_all = pd.concat([df_original_all, df], ignore_index=True)
    else:
        print(f"⚠️ File not found: {file_path}")
        
# Read all data
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
print(df_original_all.shape)



## Prepare data for prediction
check_date_str = id_date
check_time_str = id_time
text_datetime_id = '{}_{}'.format(check_date_str.replace('-', '_'), str(id_time_floored).replace(':', '_'))

# Save the data
print("Preparing data for predictions (1)...")
data_type = 'score'
collect_data(df_original_all, check_date_str, check_time_str, TEMP_DIR, 'score')

# Read the data
print("Reading data for predictions...")
df_all = pd.read_parquet(os.path.join(TEMP_DIR, f"score_{text_datetime_id}.parquet"))

print("Preparing data for predictions (1)...")
del df_original_all
df_data_02 = df_all.drop(columns = ['is_sunday'])
df_data_02['day_of_week'] = df_data_02["prediction_date_time_start"].dt.dayofweek
df_data_02['is_weekend'] = df_data_02["day_of_week"].isin([5, 6]).astype(bool)
df_data_02['h3_cell_1_month_popularity'] = df_data_02['h3_cell_1_month_popularity'].fillna(0)
df_data_02['prev_1_hour_cnt'] = df_data_02['prev_1_hour_cnt'].astype(int)
df_data_02['prev_2_hour_cnt'] = df_data_02['prev_2_hour_cnt'].astype(int)
df_data_02['prev_3_hour_cnt'] = df_data_02['prev_3_hour_cnt'].astype(int)
df_data_02['part_of_day'] = df_data_02['part_of_day'].astype(int)
df_data_02['trip_count_1_year_back'] = df_data_02['trip_count_1_year_back'].astype(int)

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




## Making predictons
print("Making the prediction")

# Select only necessary features
X_score = df_data_02[features_col]
X_score = X_score.reindex(columns=features_col, fill_value=0)

# Read model
print("Reading model...")
model_path = os.path.join(MODEL_DIR, "lgb_model.pkl")
lgb_model = joblib.load(model_path)

# Predict
print('Predicting...')
y_pred = lgb_model.predict(X_score)
print("Prediction is ready!")

# Prepare final result (as json)
df_result = df_data_02.copy()
df_result['trip_count_predict_raw'] = y_pred
df_result['trip_count_predict'] = df_result['trip_count_predict_raw'].round(0).astype(int)
df_result = df_result[['id_timestamp', 'h3_cell', 'trip_count_predict_raw', 'trip_count_predict']]

print("Saving predictions {} (json file)".format(RESULT_DIR))
df_result.to_json(os.path.join(RESULT_DIR, "prediction_{}.json".format(text_datetime_id)), orient="records", lines=True)
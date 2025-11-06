"""
collect_features.py

This module provides a feature collection function for taxi demand prediction.
It extracts features for a specific 1-hour prediction window based on historical data.

Input:
    - check_date_str: string (e.g., '2025-05-01')
    - check_time_str: string (e.g., '14:00:00')
    - data_type: 'train' or 'score' (used for output filenames)

Output:
    - Saves a parquet file to ../temp/ with extracted features and true labels (if available)

Requirements:
    - Data must be preloaded in the variable df_original_all
"""

import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
from func.part_of_day import part_of_day


def collect_data(
    df_original_sel,    # filtered historical dataframe
    check_date_str,     # e.g., "2025-05-01"
    check_time_str,     # e.g., "14:00:00"
    output_folder,      # where to save the result (aka temp train data)
    data_type = 'train' # 'train' or 'score'
):
    print("*------- Start ----------*")
    check_date_time_str = check_date_str + ' ' + check_time_str
    date_time_format = "%Y-%m-%d %H:%M:%S"
    check_date_time = datetime.strptime(check_date_time_str, date_time_format).replace(tzinfo=timezone.utc)
    print("Actual date: {}".format(check_date_time))

    floored_minute = (check_date_time.minute // 10) * 10
    select_date_time = check_date_time.replace(minute=floored_minute, second=0, microsecond=0)
    print("ID date: {}".format(select_date_time))

    # Count of the started trips in the previous windows
    df_part_1_1 = df_original_sel[
        (df_original_sel['start_date_full'] >= (select_date_time - timedelta(hours = 1))) &
        (df_original_sel['start_date_full'] < select_date_time)
    ].reset_index(drop = True)
    df_part_1_1 = df_part_1_1.groupby(by = ['h3_cell'], as_index = False)['start_date'].count().rename(columns = {'start_date': 'prev_1_hour_cnt'})
    df_part_1_1['id_timestamp'] = '{}_{}'.format(str(select_date_time.date()).replace('-', '_'), str(select_date_time.time()).replace(':', '_'))
    df_part_1_2 = df_original_sel[
        (df_original_sel['start_date_full'] >= (select_date_time - timedelta(hours = 2))) &
        (df_original_sel['start_date_full'] < (select_date_time - timedelta(hours = 1)))
    ].reset_index(drop = True)
    df_part_1_2 = df_part_1_2.groupby(by = ['h3_cell'], as_index = False)['start_date'].count().rename(columns = {'start_date': 'prev_2_hour_cnt'})
    df_part_1_2['id_timestamp'] = '{}_{}'.format(str(select_date_time.date()).replace('-', '_'), str(select_date_time.time()).replace(':', '_'))
    df_part_1_3 = df_original_sel[
        (df_original_sel['start_date_full'] >= (select_date_time - timedelta(hours = 3))) &
        (df_original_sel['start_date_full'] < (select_date_time - timedelta(hours = 2)))
    ].reset_index(drop = True)
    df_part_1_3 = df_part_1_3.groupby(by = ['h3_cell'], as_index = False)['start_date'].count().rename(columns = {'start_date': 'prev_3_hour_cnt'})
    df_part_1_3['id_timestamp'] = '{}_{}'.format(str(select_date_time.date()).replace('-', '_'), str(select_date_time.time()).replace(':', '_'))   
    df_part_1 = df_part_1_1.merge(df_part_1_2, how = 'left', on = ['h3_cell', 'id_timestamp']).merge(df_part_1_3, how = 'left', on = ['h3_cell', 'id_timestamp'])

    # Popularity of the h3 cell during last month/30 days
    df_part_2_1 = df_original_sel[
        (df_original_sel['start_date'] >= (select_date_time.date() - relativedelta(months = 1))) & 
        (df_original_sel['start_date'] < select_date_time.date())
    ]
    df_part_2_1 = df_part_2_1.groupby(by = ['h3_cell'], as_index = False)['start_date'].count()
    df_part_2_1['start_date'] = df_part_2_1['start_date'] / df_part_2_1['start_date'].max()
    df_part_2_1.rename(columns = {'start_date': 'h3_cell_1_month_popularity'}, inplace = True)
    # Popularity of the h3 cell during last week/7 days
    df_part_2_2 = df_original_sel[
        (df_original_sel['start_date'] >= (select_date_time.date() - timedelta(days = 7))) & 
        (df_original_sel['start_date'] < select_date_time.date())
    ]
    df_part_2_2 = df_part_2_2.groupby(by = ['h3_cell'], as_index = False)['start_date'].count()
    df_part_2_2['start_date'] = df_part_2_2['start_date'] / df_part_2_2['start_date'].max()
    df_part_2_2.rename(columns = {'start_date': 'h3_cell_1_week_popularity'}, inplace = True)
    df_part_2 = pd.merge(left = df_part_2_1, right = df_part_2_2, how = 'outer', on = ['h3_cell'])
    df_part_2[['h3_cell_1_month_popularity', 'h3_cell_1_week_popularity']] = df_part_2[['h3_cell_1_month_popularity', 'h3_cell_1_week_popularity']].fillna(0)
    df_part_2['id_timestamp'] = '{}_{}'.format(str(select_date_time.date()).replace('-', '_'), str(select_date_time.time()).replace(':', '_'))

    # Moving average for the same day of week during the last four weeks
    df_temp = df_original_sel[
        (
            (df_original_sel['start_date_full'] >= (select_date_time - timedelta(days = 7))) & 
            (df_original_sel['start_date_full'] <= (select_date_time - timedelta(days = 7) + timedelta(hours = 1)))
        ) | 
        (
            (df_original_sel['start_date_full'] >= (select_date_time - timedelta(days = 14))) & 
            (df_original_sel['start_date_full'] <= (select_date_time - timedelta(days = 14) + timedelta(hours = 1)))
        ) |
        (
            (df_original_sel['start_date_full'] >= (select_date_time - timedelta(days = 21))) & 
            (df_original_sel['start_date_full'] <= (select_date_time - timedelta(days = 21) + timedelta(hours = 1)))
        ) | 
        (
            (df_original_sel['start_date_full'] >= (select_date_time - timedelta(days = 28))) & 
            (df_original_sel['start_date_full'] <= (select_date_time - timedelta(days = 28) + timedelta(hours = 1)))
        ) |
        (
            (df_original_sel['start_date_full'] >= (select_date_time - timedelta(days = 35))) & 
            (df_original_sel['start_date_full'] <= (select_date_time - timedelta(days = 35) + timedelta(hours = 1)))
        ) |
        (
            (df_original_sel['start_date_full'] >= (select_date_time - timedelta(days = 42))) & 
            (df_original_sel['start_date_full'] <= (select_date_time - timedelta(days = 42) + timedelta(hours = 1)))
        ) |
        (
            (df_original_sel['start_date_full'] >= (select_date_time - timedelta(days = 49))) & 
            (df_original_sel['start_date_full'] <= (select_date_time - timedelta(days = 49) + timedelta(hours = 1)))
        )
    ]
    df_part_3 = pd.DataFrame()
    for step in [1, 2, 3, 4]:
        id_date = str(step) + '_weeks_back_moving_avg'
        df_temp_step = df_temp[
            (df_temp['start_date'] <= (select_date_time.date() - timedelta(days = 7 * step))) &
            (df_temp['start_date'] >= (select_date_time.date() - timedelta(days = 7 * (step + 3))))
        ].groupby(by = ['h3_cell', 'start_date'], as_index = False)['start_date_full'].count().rename(columns = {'start_date_full': 'trip_count'})
        df_temp_step['id_moving_average'] = id_date
        df_part_3 = pd.concat([df_part_3, df_temp_step], ignore_index = True)
        del df_temp_step
    del df_temp
    df_part_3 = df_part_3.sort_values(by = ['trip_count'], ascending = [True]).reset_index(drop = True)
    df_part_3 = df_part_3.groupby(by = ['h3_cell', 'id_moving_average'], as_index = False)['trip_count'].mean()
    df_part_3 = df_part_3.pivot(index = 'h3_cell', values = 'trip_count', columns = 'id_moving_average').reset_index(drop = False)
    df_part_3['id_timestamp'] = '{}_{}'.format(str(select_date_time.date()).replace('-', '_'), str(select_date_time.time()).replace(':', '_'))
    for step in [1, 2, 3, 4]:
        if '{}_weeks_back_moving_avg'.format(step) not in df_part_3:
            df_part_3['{}_weeks_back_moving_avg'.format(step)] = 0
    df_part_3 = df_part_3.fillna(0)
    
    # Trips count one year ago
    df_part_4 = df_original_sel[
        (df_original_sel['start_date_full'] >= (select_date_time - relativedelta(months = 12))) & 
        (df_original_sel['start_date_full'] < (select_date_time - relativedelta(months = 12) + timedelta(hours = 1)))
    ].reset_index(drop = True)
    df_part_4 = df_part_4.groupby(by = ['h3_cell'], as_index = False)['start_date'].count().rename(columns = {'start_date': 'trip_count_1_year_back'})
    df_part_4['id_timestamp'] = '{}_{}'.format(str(select_date_time.date()).replace('-', '_'), str(select_date_time.time()).replace(':', '_'))

    # Set all 3 parts into one
    df_features = pd.merge(df_part_1, df_part_3, how = 'outer', on = ['h3_cell', 'id_timestamp'])\
    .merge(df_part_2, how = 'left', on = ['h3_cell', 'id_timestamp'])\
    .merge(df_part_4, how = 'left', on = ['h3_cell', 'id_timestamp'])
    df_features['prev_1_hour_cnt'] = df_features['prev_1_hour_cnt'].fillna(0)
    df_features['prev_2_hour_cnt'] = df_features['prev_2_hour_cnt'].fillna(0)
    df_features['prev_3_hour_cnt'] = df_features['prev_3_hour_cnt'].fillna(0)
    df_features['trip_count_1_year_back'] = df_features['trip_count_1_year_back'].fillna(0)
    df_features['h3_cell_1_month_popularity'] = df_features['h3_cell_1_month_popularity'].fillna(0)
    df_features['h3_cell_1_week_popularity'] = df_features['h3_cell_1_week_popularity'].fillna(0)
    df_features[['1_weeks_back_moving_avg', '2_weeks_back_moving_avg', '3_weeks_back_moving_avg', '4_weeks_back_moving_avg']] = df_features[['1_weeks_back_moving_avg', '2_weeks_back_moving_avg', '3_weeks_back_moving_avg', '4_weeks_back_moving_avg']].fillna(0)
    df_features['prediction_date_time_start'] = select_date_time
    df_features['prediction_date_time_end'] = select_date_time + timedelta(hours = 1)
    df_features['is_weekend'] = df_features["prediction_date_time_start"].isin([5, 6]).astype(int)
    df_features['is_sunday'] = df_features["prediction_date_time_start"].isin([6]).astype(int)
    df_features['part_of_day'] = df_features['prediction_date_time_start'].apply(part_of_day)

    # Collect and save statuses
    if data_type == 'score':
        df_collect_all = df_features.copy()
    else:
        df_status = df_original_sel[
            (df_original_sel['start_date_full'] >= select_date_time) & 
            (df_original_sel['start_date_full'] < (select_date_time + timedelta(hours = 1)))
        ].reset_index(drop = True)
        df_status = df_status.groupby(by = ['h3_cell'], as_index = False)['start_date'].count().rename(columns = {'start_date': 'trip_count'})
        df_status['id_timestamp'] = '{}_{}'.format(str(select_date_time.date()).replace('-', '_'), str(select_date_time.time()).replace(':', '_'))

        # Merge features with statuses
        df_collect_all = pd.merge(df_features, df_status, how = 'outer', on = ['h3_cell', 'id_timestamp'])
        
    df_collect_all = df_collect_all[~df_collect_all['prediction_date_time_start'].isna()].reset_index(drop = True)
    df_collect_all.to_parquet(
        os.path.join(
            output_folder,
            '{}_{}_{}.parquet'.format(
                data_type,
                str(select_date_time.date()).replace('-', '_'),
                str(select_date_time.time()).replace(':', '_')
            )
        )
    )
    print("*------- End ----------*\n")
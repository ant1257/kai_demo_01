What it does:

1. Rounds the timestamp down to nearest 10-minute mark, e.g., 14:07 → 14:00.
2. Calculates lag features (how many rides started in the same H3 cell):
    in the previous 1 hour (prev_1_hour_cnt);
    in the previous 2nd hour (prev_2_hour_cnt);
    in the previous 3rd hour (prev_3_hour_cnt);
3. Calculates H3-cell popularity:
    in the last month (normalized count);
    in the last week (normalized count);
4. Calculates moving averages for this time slot from past 1–4 weeks:
    1_weeks_back_moving_avg, ..., 4_weeks_back_moving_avg;
5. Retrieves ride count for the same hour 1 year ago:
    trip_count_1_year_back;
6. Adds metadata for the current time window:
    start & end of prediction window;
    flags: is_weekend, is_sunday, part_of_day;
7. (If data_type='train'):
    counts the actual number of trips in the current prediction window -> trip_count
8. Saves everything to a .parquet file with filename format like

Example usage (pseudo-code):
for date in date_list:
    for time in check_time_list:
        collect_data(df_original_all, date, time, TEMP_DIR, data_type='train')

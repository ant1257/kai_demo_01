# streamlit_app.py
import streamlit as st
import pandas as pd
import folium
import os
from streamlit_folium import st_folium


# Streamlit settings
st.set_page_config(
    page_title="Taxi Demand Map",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Month Selector & Dynamic Loader ---
month_file_map = {
    "December 2024": "score_2024_12_all.parquet",
    "January 2025": "score_2025_01_all.parquet",
    "February 2025": "score_2025_02_all.parquet",
    "March 2025": "score_2025_03_all.parquet",
    "April 2025": "score_2025_04_all.parquet",
    "May 2025": "score_2025_05_p5_all.parquet"
}

selected_month = st.selectbox("📅 Select prediction month", list(month_file_map.keys()), index=5)
file_path = os.path.join("trip_prediction_JAN2025", "data", month_file_map[selected_month])
df = pd.read_parquet(file_path)

st.markdown(
    f"<h1 style='text-align: center;'>📈 Demonstration of predictions for {selected_month}</h1>",
    unsafe_allow_html=True
)

overall_accuracy = df['trip_count_predict_accuracy'].mean()
overall_round_accuracy = df['trip_count_predict_round_accuracy'].mean()

st.markdown(
    f"""
    <h4 style='text-align: center; font-weight: normal;'>
        Average accuracy: <b>{overall_accuracy:.2f}</b> &nbsp;|&nbsp;
        Rounded accuracy: <b>{overall_round_accuracy:.2f}</b>
    </h4>
    """,
    unsafe_allow_html=True
)

# Select timestamp
# Save and sort unique values
all_ids = sorted(df['id_timestamp'].unique())

# set session_state
if 'id_index' not in st.session_state:
    st.session_state.id_index = 0

# Reset index in case of manual selectbox selection
manual_id = st.select_slider(
    "Select id_timestamp",
    options=all_ids,
    value=all_ids[st.session_state.id_index]
)

# Reset index in session_state in case of manual selection
st.session_state.id_index = all_ids.index(manual_id)

# Next / previous button
with st.container():
    if st.button("⬅️ Previous id_timestamp"):
        if st.session_state.id_index > 0:
            st.session_state.id_index -= 1

    if st.button("➡️ Next id_timestamp"):
        if st.session_state.id_index < len(all_ids) - 1:
            st.session_state.id_index += 1

# New selected value
id_choice = all_ids[st.session_state.id_index]
df_selected = df[df['id_timestamp'] == id_choice].reset_index(drop=True)

# Map center
map_center = [14.763792, -17.352459]
m = folium.Map(location=map_center, zoom_start=12)

# Adding polygons/h3/hexagons
for _, row in df_selected.iterrows():
    polygon = row['boundary']
    hex_boundary = [(lat, lon) for lon, lat in polygon]

    # show popup on the map
    popup_content = f"A: {row['trip_count']}\nPr: {row['trip_count_predict_round']} ({row['trip_count_predict']:.2f})"

    folium.Polygon(
        locations=hex_boundary,
        color="black",
        fill=True,
        fill_opacity=0.3,
        popup=popup_content,
        tooltip=row['h3_cell']
    ).add_to(m)

# Two fields: map + information
col_map, col_info = st.columns([2, 1])

with col_map:
    st.write(f"🗺 Showing map for `{id_choice}`")
    
    avg_accuracy = df_selected['trip_count_predict_accuracy'].mean()
    avg_round_accuracy = df_selected['trip_count_predict_round_accuracy'].mean()
    st.write(f"📊 Average id_timestamp accuracy coeff.: `{avg_accuracy:.2f}`")
    st.write(f"📊 Average id_timestamp accuracy coeff. (round): `{avg_round_accuracy:.2f}`")
    
    map_data = st_folium(m, width=1000, height=800)

with col_info:
    selected_hex = map_data.get("last_object_clicked_tooltip") if map_data else None

    if selected_hex:
        row = df_selected[df_selected['h3_cell'] == selected_hex].iloc[0]
        st.subheader(f"📄 Info for H3: {selected_hex}")

        st.table(row[[
            'prediction_date_time_start',
            'prediction_date_time_end',
            'trip_count',
            'trip_count_predict',
            'trip_count_predict_accuracy',
            'trip_count_predict_round',
            'trip_count_predict_round_accuracy',
            'trip_count_1_year_back',
            'prev_1_hour_cnt',
            'prev_2_hour_cnt',
            'prev_3_hour_cnt',
            '1_weeks_back_moving_avg',
            '2_weeks_back_moving_avg',
            '3_weeks_back_moving_avg',
            '4_weeks_back_moving_avg',
            'h3_cell_1_week_popularity',
            'h3_cell_1_month_popularity',
            'is_weekend'
        ]].to_frame(name="value"))
    else:
        st.info("ℹ️ Click on a hexagon to see details.")
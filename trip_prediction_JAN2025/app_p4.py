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

# Reaad data
df = pd.read_parquet(os.path.join('.', 'data', 'score_2025_01_p4_all.parquet'))

# Выбор времени
# Сохраняем уникальные и отсортированные значения
all_ids = sorted(df['id_timestamp'].unique())

# Инициализация session_state
if 'id_index' not in st.session_state:
    st.session_state.id_index = 0

# Обновляем индекс, если выбрали вручную из selectbox
manual_id = st.select_slider(
    "Select id_timestamp",
    options=all_ids,
    value=all_ids[st.session_state.id_index]
)

# Обновляем индекс в session_state при ручном выборе
st.session_state.id_index = all_ids.index(manual_id)

# Кнопки предыдущий / следующий
with st.container():
    if st.button("⬅️ Previous id_timestamp"):
        if st.session_state.id_index > 0:
            st.session_state.id_index -= 1

    if st.button("➡️ Next id_timestamp"):
        if st.session_state.id_index < len(all_ids) - 1:
            st.session_state.id_index += 1

# Обновлённое выбранное значение
id_choice = all_ids[st.session_state.id_index]
df_selected = df[df['id_timestamp'] == id_choice].reset_index(drop=True)

# Центр карты
map_center = [14.763792, -17.352459]
m = folium.Map(location=map_center, zoom_start=12)

# Добавление гексагонов
for _, row in df_selected.iterrows():
    polygon = row['boundary']
    hex_boundary = [(lat, lon) for lon, lat in polygon]

    # Отображаем только краткий popup на карте
    popup_content = f"{row['trip_count']} ({row['trip_count_predict']:.2f})"

    folium.Polygon(
        locations=hex_boundary,
        color="black",
        fill=True,
        fill_opacity=0.3,
        popup=popup_content,
        tooltip=row['h3_cell']
    ).add_to(m)

# Два столбца: карта + информация
col_map, col_info = st.columns([2, 1])

with col_map:
    st.write(f"🗺 Showing map for `{id_choice}`")
    map_data = st_folium(m, width=1000, height=800)

with col_info:
    selected_hex = map_data.get("last_object_clicked_tooltip") if map_data else None

    if selected_hex:
        row = df_selected[df_selected['h3_cell'] == selected_hex].iloc[0]
        st.subheader(f"📄 Info for H3: {selected_hex}")

        st.table(row[[
            'prediction_date_time_start',
            'prediction_date_time_end',
            'trip_count_predict',
            'trip_count',
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
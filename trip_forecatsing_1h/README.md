** Trip Forecasting (1-Hour Ahead) **

This project provides a modular and production-ready pipeline for hour-ahead taxi demand prediction across H3 clusters. It includes training, scoring, and model management functionality.


** Project Structure **
</br>
trip_forecasting_1h:
  - code/ # Main scripts to be executed (e.g., 01_train_model.py)
  - data/ # Raw ride data in JSON format (1 file = 1 month of data)
  - func/ # Helper functions used in the scripts (e.g., feature engineering)
  - model/ # Trained model (.pkl format) + archived models and their artifacts
  - result/ # Folder intended for storing model predictions (currently empty)
  - temp/ # Temporary folder used during model training (cleared after use)


** Notes **
**code/**:  
  Contains core executable scripts. For example, `01_train_model.py` trains the forecasting model using aggregated ride data.

- **data/**:  
  Stores monthly JSON ride data. Each file corresponds to one calendar month (e.g., `data-2025-04-01.json`).

- **func/**:  
  Includes utility and helper functions used in training and preprocessing, such as feature engineering functions (`part_of_day.py`, `season_of_year.py`, etc.).

- **model/**:  
  Contains the latest trained model in `.pkl` format (for inference). Also stores previous training runs in dedicated subfolders, including the model and related artifacts (metrics, figures, etc.).

- **result/**:  
  Placeholder directory for storing model predictions. This folder is currently empty but will be populated by the prediction script.

- **temp/**:  
  Used for temporary artifacts during training (e.g., intermediate `.parquet` files). This folder is automatically cleared before and/or after each run and should not be used for persistent storage.
Taxi Demand Forecasting: 1-Hour H3-Based Model


1. Goal and Scope
The purpose of this model is to forecast taxi demand one hour ahead, using an H3 hexagonal grid.

What it predicts:
* For each H3 cell and a given 1-hour window it predicts the number of trips starting in that cell during that hour.

Where:
* The area is split into H3 cells (hexagons). Each prediction is per cell, not per individual trip.

How often to retrain:
* The model is retrained roughly once per month on the latest available data.


2. High-Level Methodology
At a high level, the pipeline consists of two main parts:
Train - uses historical trips to learn a mapping from recent history.
Prediction - applies the trained model to the most recent data to forecast demand in the next hour.

Both pipelines share the same core ideas:
* All trips are mapped to H3 cells based on their starting coordinates;
* Time is treated carefully (ISO timestamps);
* We build features that describe:
    - Recent activity;
    - Typical activity for this cell (last week/last month);
    - Patterns for this weekday and time of day (moving averages several weeks back);
    - Seasonality (similar time one year ago);
    - Calendar context (day of week, part of day);

The model itself is a gradient boosting regressor (LightGBM), which is well-suited for tabular data and heterogeneous features (https://en.wikipedia.org/wiki/LightGBM).


3. Training Pipeline
* Data Used for Training
The training pipeline works on historical trip data with the following key fields available per trip:
    - Trip start timestamp;
    - Starting latitude and longitude (H3 cell);
Every trip is assigned to an H3 cell based on its starting coordinates, so we can aggregate data by a cell.

* What is Being Trained
A regression model where:
    - Input (features) describes the state of each H3 cell at a specific hour;
    - Target (label) is the number of trips that actually started in that cell during the next hour;
We generate many examples by sliding over time (in 10-minute steps) and over all H3 cells WITH ACTIVITY.

* Key Feature Groups
Recent activity (these features capture how busy the cell has been in the last few hours).
Cell popularity (these features describe how "popular" a cell is over longer periods).
Weekly pattern (moving averages for same weekday/time).
Seasonality (number of trips in this H3 cell in a similar 1-hour window roughly one year ago.).
Calendar context (we also include basic calendar and time-of-day indicators).

* Model and Evaluation
We use a LightGBM Regressor, a gradient boosting model optimized for speed and performance on tabular data.

During training we:
    - Split the data into train and test sets;
    - Train the model on the train set;
    - Evaluate on the test set (Mean Absolute Error, Root Mean Squared Error, Coefficient of Determination);


4. Prediction Pipeline
At prediction time, we want to answer: "Given all trips and activity up to now, how many trips do we expect in each H3 cell in the next hour?"
To do that, the prediction pipeline:
    - Takes the current timestamp (rounded down to the nearest 10 minutes);
    - Collects recent historical trips up to that time;
    - Rebuilds the same features as in training;
    - Applies the trained LightGBM model to this feature set;
    - Produces, for each H3 cell a raw prediction (continuous value), and a rounded prediction (integer trip count);

Output of the Prediction Pipeline:
[PLEASE CHANGE THIS PART, BECAUSE THE APPPROACH HAS PAROBABLY CHANGED!!!!]
For each scoring run, the model outputs a table (or JSON) with:
    - Time window identifier;
    - H3 cell ID;
    - Predicted trip count;


5. Summary
The main idea to keep in mind is: "We are not predicting individual trips, we are predicting how many trips will start in each hexagonal cell of the city in the next hour, using a structured set of features derived from historical data."
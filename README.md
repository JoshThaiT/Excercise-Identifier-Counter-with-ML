# Exercise Counter using Computer Vision

Upload short form video which identifies exercises and will count the amount of reps using pose estimation and machine learning

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://blank-app-template.streamlit.app/](https://blank-app-wz1wvxismjo.streamlit.app/))

## Overview
This project uses computer vision, specifically Yolopose, to detect human poses and will identify 3 types of exercises using machine learning.
Pull Ups, Push Ups and Sit Ups.
The repition of each exercise will be counted using a state machine based of exercise biomechanics.

This project was built to explore real-time ML inference and exercise tracking.

## Demo
([App Demo](https://github.com/JoshThaiT/Excercise-Identifier-Counter-with-ML/blob/main/Picture1.gif))

## Features
- Real time pose detection
- Automatic repetition counting

## Tech Stack
- OpenCV
- Streamlit
- Imageio
- YoloPose
- XGBoost / LSTM

## Install yourself
1. Install requirements
   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

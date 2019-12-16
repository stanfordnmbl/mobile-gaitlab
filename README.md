# Extracting gait metrics from videos using convolutional neural networks

Implementation of algorithms for:
"Clinical gait analysis at home: Deep neural networks enable quantitative movement analysis using single-camera videos"
by Łukasz Kidziński*, Bryan Yang*, Jennifer Hicks, Apoorva Rajagopal, Scott Delp, Michael Schwartz

This code requires data (~0.5GB), currently available on request. Please contact lukasz.kidzinski@stanford.edu

## Contents

| File | Description |
|:------------- |:-------------|
| process_annotations.ipynb | Processes OpenPose json files |
| process_frames.ipynb | Normalize pozes in frames |
| combine_video_csvs.ipynb | Combines time series of poses with labels |
| split_ids.ipynb | Training, validation, testing split |
| compute_SEMLS_residuals.ipynb | Builds simple models for SEMLS to control for demographics |
| process_raw_videos.ipynb | Additional processing of trajectories (missing data, smoothing, orientation) |
| cnn_predict_doublesided_var.ipynb | Models for variables that depend on the side (GDI, knee flexion at max extension) |
| cnn_predict_singlesided_var.ipynb | Models for variables that don't depend on the side (cadence, speed) |
| select_optimal_epoch.ipynb | Choose the best model based on validation error |
| calculate_corr_rmse.ipynb | Get performance metrics for classification and regression tasks |
| calculate_SEMLS_ROC.ipynb | Get performance metrics for the binary prediction task (SEMLS) |
| statistical_analysis.ipynb | Analyze results from models |

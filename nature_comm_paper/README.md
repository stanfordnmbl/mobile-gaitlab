# Training and the analysis

For training our models you need to install `requirements.txt` from the main directory
```bash
pip install -r requirements.txt
```
Data used for training is available [here](https://simtk.org/frs/?group_id=1918) (doi:10.18735/j0rz-0k12)

Use the following scripts for training and analyzing models. Files are listed in the order in which they should be run.

| File | Description |
|:------------- |:-------------|
| process_annotations.ipynb* | Processes OpenPose json files |
| process_frames.ipynb* | Normalize pozes in frames |
| combine_video_csvs.ipynb | Combines time series of poses with labels |
| split_ids.ipynb | Training, validation, testing split |
| compute_SEMLS_residuals.ipynb | Builds simple models for SEMLS to control for demographics |
| process_raw_videos.ipynb* | Additional processing of trajectories (missing data, smoothing, orientation) |
| cnn_predict_doublesided_var.ipynb* | Models for variables that depend on the side (GDI, knee flexion at max extension) |
| cnn_predict_singlesided_var.ipynb* | Models for variables that don't depend on the side (cadence, speed) |
| select_optimal_epoch.ipynb | Choose the best model based on validation error |
| calculate_corr_rmse.ipynb | Get performance metrics for classification and regression tasks |
| calculate_SEMLS_ROC.ipynb | Get performance metrics for the binary prediction task (SEMLS) |
| statistical_analysis.ipynb | Analyze results from models |

For predicting on new video data you need to run files with *.

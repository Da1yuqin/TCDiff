# Trajectory Model README

This is a lightweight trajectory prediction model.

Runs efficiently on a single GPU; typical training takes several hours to days depending on dataset size.

Current model predicts only XY coordinates, using a sliding window of 100 frames and a step of 25, which was empirically found to outperform other settings like 60/30.

Train the model using the following commands:

    python train_traj.py --device cuda:0 
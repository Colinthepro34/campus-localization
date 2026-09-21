"""
PLACEHOLDER -- not implemented. Future work per project spec section 43.
Intended design once single-image inference (Task 14) is proven reliable:
  1. Capture frames from a live camera feed at a fixed interval.
  2. Run predict_image() (from src/inference/predict.py) on each frame.
  3. Apply temporal smoothing (e.g. majority vote or exponential smoothing
     of the probability vector) across the last N frames to avoid flicker.
  4. Overlay the smoothed location + floor label on the video feed.
Do not implement this until Task 14's single-image accuracy and
confidence calibration have been reviewed by a human.
"""
raise NotImplementedError("Future work -- see project spec section 43 and this file's docstring.")
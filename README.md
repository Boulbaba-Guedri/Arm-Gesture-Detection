# Arm-Gesture-Detection

**The Idea:** Real-time arm gesture detection algorithm using a camera.

# Real-Time Arm Gesture Detection Algorithm with Smart Camera

This algorithm enables the real-time detection and interpretation of user arm gestures using a camera. It's designed for intuitive human-machine interaction applications, such as remote control, virtual reality, and assistance for individuals with limited mobility.

# Features

* Real-time arm gesture detection.
* Recognition of a predefined set of gestures (e.g., raise arm, lower arm, etc.).
* Calculation of arm angles and positions.
* Adaptation to varying lighting conditions.
* Optimization for low latency.

# Architecture

The algorithm consists of the following steps:

1.  **Image Acquisition:** The smart camera captures a video stream.
2.  **Image Preprocessing:** Images are processed to enhance quality and reduce noise.
3.  **Arm Detection:** Applies our image processing ideas based on fuzzy logic and CNN.
4.  **Pose Estimation:** Estimates the key points of the arms.
5.  **Gesture Recognition:** Interprets arm movements and recognizes gestures.
6.  **Output:** Recognized gestures are displayed on the monitoring window.

# Technical Details

1. **Models Used:** The SCT algorithms consist of two algorithms: one for SCT-O and the other for SCT-T.
2. *Libraries Used:** OpenCV for the SCT-O model and TensorFlow for SCT-T.
3. *Programming Language:** Python.
4. *Program File Names:** `ok6.1.py` (for SCT-O model) and `ok6.2.py` (for SCT-T model).
=> The `ok6.1.py` and `ok6.2.py` files are extensively commented to facilitate source code understanding.
5. *Hardware Requirements:** Surveillance camera/USB camera/integrated camera, computer with GPU (recommended).

# Contribution

Contributions are welcome! Please submit a pull request with your improvements.

# Licence

Ce projet est distribué sous la licence MIT.

# Corresponding Author : 

boulbaba.guedri@ensit.u-tunis.tn (Boulbaba Guedri),naji.guedri@ensit.u-tunis.tn (Naji Guedri), rached.gharbi@ensit.rnu.tn (Rached Gharbi).


# Motion-Tracking-on-Video-using-OpenCV

This project uses OpenCV to perform **real-time motion tracking** in a video file (`GTA 6.mp4`) using **background subtraction** with the MOG2 algorithm.

---

### 🛠️ Features

* Detects motion in a video using background subtraction.
* Draws bounding boxes around moving objects.
* Ignores small movements to reduce noise.
* Visualizes both the original frame with motion boxes and the foreground mask.

---

### 🖼️ Example Output

| Motion Tracking                                | Foreground Mask                                |
| ---------------------------------------------- | ---------------------------------------------- |
| ![Image](https://github.com/user-attachments/assets/3d8b4c62-8952-4a6f-8392-1240954ae272)| ![Image](https://github.com/user-attachments/assets/c9976811-f7da-4858-be28-bc5e4c9fbc51)|


### 📁 Installation

```bash
git clone https://github.com/your-username/motion-tracking-opencv.git
cd motion-tracking-opencv
pip install opencv-python
```

---

### ▶️ How to Run

Make sure you have a video file named `Sample.mp4` in the same directory. Then run:

```bash
python motion_tracking.py
```

> ✅ You can also replace `'Sample.mp4'` with `0` to use your webcam instead.

---

### 📄 Code Overview

```python
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100)
```

* Initializes the background subtractor with parameters for sensitivity.

```python
cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

* Detects contours in the foreground mask.

```python
cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

* Draws bounding boxes around detected moving objects.

---

### 🚪 Exit

Press `Esc` (Escape key) while the window is active to stop the video and close all OpenCV windows.

---

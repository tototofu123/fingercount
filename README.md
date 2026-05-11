# Finger Count Recognizer ✋🔢

A machine learning project that uses computer vision to recognize and count fingers in real-time via your webcam.

## 🚀 Features
- **Real-time Tracking**: Uses high-speed pose estimation to track hand movements.
- **Accurate Counting**: Detects extended vs. folded fingers to provide a precise count (0-5).
- **Interactive UI**: Visual skeleton overlay and real-time counter display.

## 🛠️ Tech Stack
- **Python 3**
- **OpenCV** (Computer Vision library)
- **MediaPipe** (Hand tracking framework)

## 📂 Project Structure
- `finger_counter.py`: Main execution script for camera capture and detection logic.
- `.gitleaks.toml`: Integrated secret scanning for repository safety.
- `site/`: Contains web-based documentation and demos.

## 📖 How to Run
1. Install dependencies:
```bash
pip install opencv-python mediapipe
```
2. Run the counter:
```bash
python finger_counter.py
```

## 📜 License
MIT

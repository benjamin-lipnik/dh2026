# 🥊 Multiplayer Boxing Game with Computer Vision Controls

This project is a multiplayer boxing game built in Unity, controlled using real-time computer vision. Players use their body movements, captured via a webcam, to perform in-game actions such as punches and gestures.

The system consists of two main components:
- A Python-based pose detection system using MediaPipe
- A Unity-based multiplayer game that receives movement data over UDP

---

## 📦 Project Overview

1. **Python Pose Detection (Computer Vision)**
   - Captures webcam feed
   - Uses MediaPipe to estimate full-body pose
   - Interprets body posture into game gestures (e.g., punches, blocks)
   - Sends gesture data via UDP socket to Unity

2. **Unity Game Client**
   - Receives gesture data from Python over UDP
   - Maps detected gestures to in-game actions
   - Handles multiplayer boxing gameplay logic

---

## ⚠️ Important

The pose detection system and the Unity game are **separate applications**.

👉 You must run the Python detection software **independently** from the Unity game.

Without the Python process running, the game will not receive any input data.

---

## ▶️ How to Run

### 1. Start Pose Detection (Python)
Navigate to the Python project folder and run:

```bash
python main.py

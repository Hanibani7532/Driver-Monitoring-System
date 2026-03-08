A comprehensive Computer Vision solution designed to enhance road safety by detecting driver fatigue and monitoring on-road objects in real-time.

🌟 Key Features
Drowsiness Detection: Real-time monitoring of eye states to detect fatigue and trigger instant alerts.

Object Detection: Integrated with YOLOv8 for identifying potential hazards or objects in the driver's path.

Modular Architecture: Separate modules for data preparation, object detection, and drowsiness analysis.

High Performance: Optimized inference pipelines for low-latency feedback.

🛠️ Tech Stack
Language: Python

Computer Vision: OpenCV, Cvzone, MediaPipe

Deep Learning: YOLOv8 (Ultralytics)

Domain: Artificial Intelligence, Road Safety

📂 Project Structure
Data/: Contains raw and processed datasets.

Models/: Storage for trained models and weights (including yolov8n.pt).

Scripts/:

drowsiness_detect.py: Logic for monitoring eye closure and fatigue.

object_detect.py: YOLOv8 implementation for object recognition.

main_inference.py: Primary script to run the combined monitoring system.

prepare_data.py: Data preprocessing and augmentation utilities.

final_dms.py: The final integrated application script.

## 📊 Dataset
The dataset used for training and testing is too large for GitHub. You can download it from the following source:
* **Source:** [State Farm Distracted Driver Detection (Kaggle)](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data)

🚀 Getting Started
1. Installation
Clone the repository and install the required dependencies:

Bash
git clone https://github.com/Hanibani7532/Driver-Monitoring-System.git
pip install -r requirements.txt
2. Run the System
To start the real-time driver monitoring system, run:

Bash
python final_dms.py
👤 Author
Muhammad Hanzala

AI/ML Engineer

BS in Artificial Intelligence, COMSATS University
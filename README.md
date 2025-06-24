# 🎓 Student Face Recognition System

This project implements a real-time face recognition system for student identification using CCTV cameras. It leverages **InsightFace** for face detection and embedding, **FAISS** for fast similarity search, and **MySQL** to log detections. The system supports up to 4 live camera streams and can track and log students' appearances in real-time.

## 📁 Project Structure

- `training.ipynb` – Extracts facial embeddings from the dataset and saves:
  - `face_index.faiss` – FAISS index of embeddings.
  - `labels.pkl` – Corresponding student names/IDs.

- `live.py` – Performs real-time face detection and recognition from one or more CCTV cameras using:
  - FAISS to identify faces.
  - OpenCV for camera stream and display.
  - SQL logger to save timestamps of recognized faces.

- `db.py` – Manages MySQL database operations for logging detections with timestamp, camera ID, and student roll number.

## 🔧 Features

✅ Real-time multi-camera face recognition  
✅ FAISS-based fast face matching  
✅ InsightFace for accurate facial embeddings  
✅ MySQL logging with student name, confidence score, and timestamp  
✅ Intelligent duplicate detection prevention (1-minute interval)  
✅ FPS display and multi-camera grid UI

## 🛠️ Requirements

- Python 3.7+ (3.10 preferred)
- CUDA-compatible GPU (for InsightFace with GPU acceleration)
- Libraries:
  - `opencv-python`
  - `faiss-gpu`
  - `insightface`
  - `numpy`
  - `mysql-connector-python`
  - `pickle`

Install dependencies:

```bash
pip install -r requirements.txt
```

## 💾 MySQL Setup

Make sure to have a MySQL server running. Create the database `face_db`:

```sql
CREATE DATABASE face_db;
```

The required table will be auto-created by `db.py` when `live.py` is run.

## 📹 Demo
### Video 

https://github.com/user-attachments/assets/f793810b-813b-4fc4-a40f-e1a146b22ae4


People who are in known datasets are boxxed green and are updated with their appearence time in db and red boxxed people are unknown people.

### Website
http://sync-cv-presentation-ui.vercel.app

## 🚀 Usage

### 1. Train the System

Extract embeddings from the student face dataset:

```bash
# Open and run the cells in the notebook
training.ipynb
```

This generates:
- `face_index.faiss` – FAISS index of embeddings
- `labels.pkl` – Mapping of embeddings to names

### 2. Run Live Detection

Start real-time recognition from CCTV feed (default camera 0):

```bash
python live.py
```

To add more cameras, edit the `camera_urls` list in `live.py`:

```python
camera_urls = [
    0,  # Default webcam
    'rtsp://192.168.1.2:554/live',  # Example IP Camera
    ...
]
```

### 3. View Logs

All detections are saved in the MySQL database in the `detections` table:
- `roll_no`
- `confidence`
- `timestamp`
- `camera`

## 📂 Example Output

- Bounding boxes with labels on live video feed.
- Real-time FPS display.
- Console logs for matches and errors.

## ⚠️ Notes

- Make sure to install GPU drivers and CUDA to leverage real-time processing.
- Embedding quality is crucial. Use high-resolution, clear face images during training.

## 👨‍💻 Author

- Prabanand S C - https://github.com/scprabanand
- Mohammed Afeef M - https://github.com/afeefm05
- Raghul S - https://github.com/Raghulskr12
- Suresh S U - https://github.com/SURESH-S-U
- Dinesh R - https://github.com/dineshdinz12
- Mohith S - https://github.com/MOHITH2511
- Jhai Pranesh T - https://github.com/Jhai-pranesh
- Jisnu S - https://github.com/Jisnu-Dev


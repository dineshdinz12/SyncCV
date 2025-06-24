# SyncCV: Student Face Recognition System

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Open Issues](https://img.shields.io/github/issues/dineshdinz12/SyncCV)](https://github.com/dineshdinz12/SyncCV/issues)

A robust, real-time face recognition system for student identification using CCTV cameras. Built with **InsightFace** for face detection, **FAISS** for fast similarity search, and **MySQL** for logging. Supports up to 4 live camera streams, with intelligent duplicate detection prevention and a modern multi-camera UI.

---

## 📑 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [MySQL Setup](#mysql-setup)
- [Usage](#usage)
- [Demo](#demo)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)

---

## 🚀 Features
- **Real-time multi-camera face recognition** (up to 4 streams)
- **FAISS-based fast face matching** for large datasets
- **InsightFace** for state-of-the-art facial embeddings
- **MySQL logging**: student name, confidence, timestamp, camera
- **Duplicate detection prevention** (1-minute interval)
- **Live FPS display** and multi-camera grid UI
- **Easy extensibility** for new cameras or datasets

---

## 📁 Project Structure
- `training.ipynb` – Extracts facial embeddings from your dataset and saves:
  - `face_index.faiss` – FAISS index of embeddings
  - `labels.pkl` – Student names/IDs
- `live.py` – Real-time face detection and recognition from CCTV cameras
- `db.py` – MySQL database operations for logging detections
- `requirements.txt` – Python dependencies

---

## 🛠️ Requirements
- Python 3.7+ (3.10 recommended)
- CUDA-compatible GPU (for best performance with InsightFace)
- MySQL server (local or remote)
- Python libraries:
  - `opencv-python`
  - `faiss-gpu` (or `faiss-cpu` for non-Linux)
  - `insightface`
  - `numpy==1.24.4`
  - `mysql-connector-python`
  - `pickle-mixin`
  - `onnxruntime`
  - `torch` (version matching your CUDA)

---

## 📦 Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/dineshdinz12/SyncCV.git
   cd SyncCV
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   > **Note:** Adjust `faiss-gpu` to `faiss-cpu` if not using Linux, and set the correct `torch` version for your CUDA.

---

## 💾 MySQL Setup
1. **Start your MySQL server.**
2. **Create the database:**
   ```sql
   CREATE DATABASE face_db;
   ```
3. **Table creation:**
   The required `detections` table is auto-created by `db.py` when you run `live.py`.
   - Table schema:
     | Column     | Type           | Description                |
     |------------|----------------|----------------------------|
     | id         | INT, PK, AI    | Detection ID               |
     | roll_no    | VARCHAR(100)   | Student roll number/name   |
     | confidence | FLOAT          | Recognition confidence     |
     | timestamp  | DATETIME       | Detection time             |
     | camera     | INT            | Camera ID                  |

---

## 🧑‍💻 Usage

### 1. Train the System
- Prepare a dataset: Each subfolder in your dataset directory should be named after the student (roll number or name) and contain their face images.
- Run the notebook:
  ```bash
  # Open and execute all cells in
  training.ipynb
  ```
- This generates:
  - `face_index.faiss` (embeddings index)
  - `labels.pkl` (mapping of embeddings to names)

### 2. Run Live Detection
- Start real-time recognition from the default camera:
  ```bash
  python live.py
  ```
- **To add more cameras:**
  Edit the `camera_urls` list in `live.py`:
  ```python
  camera_urls = [
      0,  # Default webcam
      'rtsp://192.168.1.2:554/live',  # Example IP Camera
      # Add more camera URLs or indices as needed
  ]
  ```

### 3. View Logs
- All detections are saved in the MySQL `detections` table.
- You can query logs using any MySQL client:
  ```sql
  SELECT * FROM detections ORDER BY timestamp DESC;
  ```

---

## 🎬 Demo

<video src="https://github.com/user-attachments/assets/f793810b-813b-4fc4-a40f-e1a146b22ae4" controls width="600"></video>

> **Legend:**
> - Green box: Known student (logged in DB)
> - Red box: Unknown person

---

## 🧩 Troubleshooting
- **CUDA/torch errors:** Ensure your GPU drivers and CUDA are installed, and `torch` matches your CUDA version.
- **MySQL connection issues:** Check your MySQL server is running and credentials in `db.py` are correct.
- **No face detected:** Use high-quality, front-facing images for training.
- **Performance:** For best speed, use a CUDA GPU and keep the number of live streams ≤ 4.

---

## 🤝 Contributing
1. Fork this repo and create your feature branch (`git checkout -b feature/YourFeature`)
2. Commit your changes (`git commit -am 'Add new feature'`)
3. Push to the branch (`git push origin feature/YourFeature`)
4. Open a Pull Request

---

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👨‍💻 Authors
- [Prabanand S C](https://github.com/scprabanand)
- [Mohammed Afeef M](https://github.com/afeefm05)
- [Raghul S](https://github.com/Raghulskr12)
- [Suresh S U](https://github.com/SURESH-S-U)
- [Dinesh R](https://github.com/dineshdinz12)
- [Mohith S](https://github.com/MOHITH2511)
- [Jhai Pranesh T](https://github.com/Jhai-pranesh)
- [Jisnu S](https://github.com/Jisnu-Dev)

---

> Made with ❤️ by the SyncCV Team


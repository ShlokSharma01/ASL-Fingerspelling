# Real-Time ASL Fingerspelling Detection using Deep Learning

## 📌 Overview
This project implements a **real-time American Sign Language (ASL) fingerspelling detection system** using **deep learning and computer vision**.  
It detects static ASL alphabet letters (A–Z) from a webcam feed and displays the predicted letter in real-time.

The model is trained using the **ASL Alphabet Dataset** from Kaggle and leverages **ResNet50** with transfer learning for high accuracy.

---

## 🚀 Features
- **Real-time detection** from webcam
- **26 classes** (A–Z fingerspelling signs)
- **GPU acceleration** for faster training and inference
- **Transfer learning** with ResNet50
- **User-friendly interface** with ROI guidance
- **100% test accuracy** achieved during training

---

## 📂 Dataset
This project uses the **[ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)**.  
Download and extract the dataset into the `dataset/` folder:
dataset/
A/
B/
C/
...
Z/

> ⚠ The dataset is **not included** in this repository due to size limits. You must download it manually.

---

## 🛠️ Requirements
- Python 3.7–3.11
- PyTorch
- Torchvision
- OpenCV
- Pillow
- NVIDIA GPU (optional but recommended for speed)

Install dependencies:
```bash
pip install -r requirements.txt


ASL-Fingerspelling/
│── dataset/                # Dataset folder (A–Z images)
│── scripts/
│   ├── train.py             # Train the ResNet50 model
│   ├── predict_webcam.py    # Real-time detection script
│── asl_model.pth            # Trained model weights (generated after training)
│── requirements.txt         # Python dependencies
│── README.md                # Project documentation

📖 How to Use
1️⃣ Train the Model
bash
Copy
Edit
python scripts/train.py
Splits dataset into train/ and test/

Trains a ResNet50 model for 30 epochs

Saves model as asl_model.pth

2️⃣ Run Real-Time Detection
bash
Copy
Edit
python scripts/predict_webcam.py
Opens webcam feed

Place your hand inside the green ROI box

Displays the detected ASL letter in real-time

Press Q to quit

📊 Results
Test Accuracy: 100% on held-out dataset

Real-time detection works smoothly on both CPU and GPU

Optimized data augmentation for better generalization

💡 Future Improvements
Add dynamic sign recognition (continuous signing)

Integrate into a mobile app

Build a full sign-to-text translator

Support more sign language variants

📜 License
This project is for educational purposes only. Dataset belongs to the original creator on Kaggle.

🙏 Acknowledgments

Kaggle - ASL Alphabet Dataset
PyTorch
OpenCV
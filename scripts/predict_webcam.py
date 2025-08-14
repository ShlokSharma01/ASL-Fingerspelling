# scripts/predict_webcam.py
import cv2
import torch
from torchvision import transforms, models
from PIL import Image

# === CONFIG ===
MODEL_PATH = "asl_model.pth"
CLASS_NAMES = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')  # Update to match dataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD MODEL ===
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === WEBCAM LOOP ===
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define ROI - Appropriate size for 128px model
    roi_x1, roi_y1, roi_x2, roi_y2 = 100, 100, 356, 356

    # Draw ROI rectangle
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 3)
    cv2.putText(frame, 'Place hand here', (roi_x1 + 10, roi_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Extract ROI
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    if roi.size > 0:
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            label = CLASS_NAMES[predicted.item()]

        cv2.putText(frame, f"Prediction: {label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("ASL Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
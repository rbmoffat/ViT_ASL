import cv2
import mediapipe as mp
import torch
import torch.nn.functional as F
from torchvision import transforms
from vit_model import ASLVisionTransformer, ASL_LABELS
import time
from collections import deque, Counter
import os
import numpy as np

# Set to 1 to enable saving images and video
save_images = 0
save_video = 1

# Create outputs directory if it doesn't exist
if save_images or save_video:
    os.makedirs('../outputs', exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Load the ASL model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASLVisionTransformer(num_classes=len(ASL_LABELS))
model.load_state_dict(torch.load('../outputs/asl_vit_model.pth', map_location=device))
model.to(device)
model.eval()

# Define the transform for the model input
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_hand_image(image, hand_landmarks, padding_factor=1.5):
    h, w, _ = image.shape
    
    # Get hand bounding box
    x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
    x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
    y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
    y_max = max([lm.y for lm in hand_landmarks.landmark]) * h
    
    # Calculate center of the hand
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    
    # Calculate the size of the square
    size = max(x_max - x_min, y_max - y_min) * padding_factor
    size = min(size, min(h, w))  # Ensure size doesn't exceed image dimensions
    
    # Calculate square boundaries
    left = int(max(0, center_x - size / 2))
    top = int(max(0, center_y - size / 2))
    right = int(min(w, left + size))
    bottom = int(min(h, top + size))
    
    # Extract and resize the hand image
    hand_image = image[top:bottom, left:right]
    hand_image = cv2.resize(hand_image, (224, 224))
    
    assert hand_image.shape[:2] == (224, 224), f"Image shape is {hand_image.shape[:2]}, expected (224, 224)"
    
    return hand_image, (left, top, right, bottom)

def predict(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
    
    _, predicted = torch.max(output, 1)
    predicted_label = ASL_LABELS[predicted.item()]
    
    return predicted_label

# Initialize the camera
cap = cv2.VideoCapture(0)

# Get video properties for saving
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30  # Set constant frame rate for output video

# Initialize video writer
if save_video:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('../outputs/output_video.avi', fourcc, fps, (frame_width, frame_height))

# Variables for prediction timing
last_prediction_time = 0
last_model_run_time = 0
prediction_interval = 0.5  # 0.5 seconds (2 times per second)
model_run_interval = 0.1  # 0.1 seconds (10 times per second)
current_prediction = ""
prediction_window = deque(maxlen=5)  # Store last 5 predictions (0.5 seconds worth)

frame_count = 0
start_time = time.time()
last_frame_time = start_time

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        current_time = time.time()
        frame_time = current_time - last_frame_time
        last_frame_time = current_time

        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_image, (left, top, right, bottom) = extract_hand_image(rgb_image, hand_landmarks)
                
                if current_time - last_model_run_time >= model_run_interval:
                    prediction = predict(hand_image)
                    prediction_window.append(prediction)
                    last_model_run_time = current_time
                    
                    if save_images:
                        cv2.imwrite(f'../outputs/frame_{int(current_time*1000)}.jpg', cv2.cvtColor(hand_image, cv2.COLOR_RGB2BGR))
                
                if current_time - last_prediction_time >= prediction_interval:
                    current_prediction = Counter(prediction_window).most_common(1)[0][0]
                    last_prediction_time = current_time
                
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(image, f"Prediction: {current_prediction}", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('ASL Recognition', image)
        
        if save_video:
            elapsed_time = current_time - start_time
            expected_frames = int(elapsed_time * fps)
            frames_to_write = max(1, int(frame_time * fps))
            
            for _ in range(frames_to_write):
                if frame_count <= expected_frames:
                    out.write(image)
                    frame_count += 1

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cap.release()
    if save_video:
        out.release()
    cv2.destroyAllWindows()

if save_video and os.path.exists('../outputs/output_video.avi'):
    print(f"Video file created. Size: {os.path.getsize('../outputs/output_video.avi')} bytes")

if save_images:
    frame_images = [f for f in os.listdir('../outputs') if f.startswith('frame_') and f.endswith('.jpg')]
    if frame_images:
        print(f"Created {len(frame_images)} individual frame images")
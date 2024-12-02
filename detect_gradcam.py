import datetime
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torch.nn as nn

# define some constants
CONFIDENCE_THRESHOLD = 0.2
GREEN = (0, 255, 0)

names = ['auto', 'bicycle', 'bike', 'board', 'building', 'bus', 'car', 'cars', 'divider', 'milestone', 'motor', 
         'motorbike', 'person', 'rider', 'signboard', 'stop sign', 'street light', 'tower', 'tractor', 
         'traffic light', 'tree', 'truck', 'van']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained YOLOv8n model
model = YOLO("E:\\Study\\SEM 7\\xai\\Project\\best.pt")
model.to(device)

# Grad-CAM Hook Registration
gradients = None

def save_gradients(grad):
    global gradients
    gradients = grad

# Register hooks on the specific layer (e.g., the last convolutional layer)
def get_last_conv_layer(model):
    # Assuming the model has a module called 'model', and it ends with a convolutional layer
    for name, layer in model.model.named_modules():
        if isinstance(layer, nn.Conv2d):
            last_conv_layer = layer
    return last_conv_layer

last_conv_layer = get_last_conv_layer(model)
last_conv_layer.register_backward_hook(save_gradients)

# Function to apply Grad-CAM
def apply_gradcam(feature_maps):
    global gradients
    
    # Pool the gradients across the height and width dimensions
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    
    # Multiply the feature maps by the pooled gradients to get the weighted feature maps
    for i in range(pooled_gradients.size(0)):
        feature_maps[:, i, :, :] *= pooled_gradients[i]
    
    # Average the weighted feature maps along the channel dimension to obtain the Grad-CAM heatmap
    heatmap = torch.mean(feature_maps, dim=1).squeeze()
    
    # Apply ReLU to the heatmap to only keep positive values
    heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)
    
    # Normalize the heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    return heatmap

# Initialize the video capture object
cap = cv2.VideoCapture("E:\\Study\\SEM 7\\xai\\Project\\curved_lane.mp4")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Resize the frame to 640x640 (or another size divisible by 32)
    resized_frame = cv2.resize(frame, (640, 640))

    # Convert the resized frame to a tensor
    input_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(device)

    # Get predictions from the model
    detections = model(input_tensor)[0]

    # # Convert the frame to a tensor and run it through the model
    # input_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    # input_tensor = input_tensor.to(device)

    # # Get predictions from the model
    # detections = model(input_tensor)[0]

    # Loop over the detections and apply Grad-CAM
    for data in detections.boxes.data.tolist():
        confidence = data[4]

        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        class_id = int(data[5]) if len(data) == 6 else int(data[6])

        # Backpropagate the score for the detected class
        model.zero_grad()
        target = detections.boxes[:, 5] == class_id
        target = target.float().mean().to(device)
        target.backward(retain_graph=True)

        # Get the feature maps from the last convolutional layer
        feature_maps = last_conv_layer.output

        # Apply Grad-CAM to obtain the heatmap
        heatmap = apply_gradcam(feature_maps)

        # Resize the heatmap to the size of the frame
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

        # Convert the heatmap to an RGB image
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Overlay the heatmap on the original frame
        overlay_frame = heatmap * 0.4 + frame

        # Display the frame with the Grad-CAM overlay
        cv2.imshow("Grad-CAM", overlay_frame)
        
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

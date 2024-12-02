import datetime
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torch.nn as nn

# Define some constants
CONFIDENCE_THRESHOLD = 0.0
CAR_CLASS_ID = 6  # Assuming the index for 'car' in your `names` list is 6
GREEN = (0, 255, 0)

names = ['auto', 'bicycle', 'bike', 'board', 'building', 'bus', 'car', 'cars', 'divider', 'milestone', 'motor', 
         'motorbike', 'person', 'rider', 'signboard', 'stop sign', 'street light', 'tower', 'tractor', 
         'traffic light', 'tree', 'truck', 'van']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained YOLOv8n model
model = YOLO("yolov8n.pt")
model.to(device)

# Grad-CAM++ Hook Registration
gradients = None
activations = None

def save_gradients(grad):
    global gradients
    gradients = grad

def save_activations(act):
    global activations
    activations = act

# Register hooks on the specific layer (e.g., the last convolutional layer)
def get_last_conv_layer(model):
    for name, layer in model.model.named_modules():
        if isinstance(layer, nn.Conv2d):
            last_conv_layer = layer
    return last_conv_layer

last_conv_layer = get_last_conv_layer(model)
last_conv_layer.register_forward_hook(lambda module, input, output: save_activations(output))
last_conv_layer.register_backward_hook(lambda module, grad_input, grad_output: save_gradients(grad_output[0]))

# Function to apply Grad-CAM++
def apply_gradcam_plus():
    global gradients, activations

    # Calculate weights for GradCAM++
    alpha_numerators = gradients.pow(2)
    alpha_denominators = 2 * gradients.pow(2) + torch.sum(activations * gradients.pow(3), dim=[2, 3], keepdim=True)
    alpha_denominators = torch.where(alpha_denominators != 0, alpha_denominators, torch.ones_like(alpha_denominators))
    alphas = alpha_numerators / alpha_denominators

    # Aggregate the alphas
    weights = torch.sum(alphas * torch.relu(gradients), dim=[2, 3])

    # Calculate the Grad-CAM++ heatmap by multiplying the weights by activations
    heatmap = torch.sum(weights.unsqueeze(2).unsqueeze(3) * activations, dim=1).squeeze()

    # Apply ReLU to the heatmap to only keep positive values
    heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)

    # Normalize the heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return heatmap

# Initialize the video capture object
cap = cv2.VideoCapture("E:\\Study\\SEM 7\\xai\\Project\\curved_lane.mp4")

# Get the original frame width and height
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

output_video = cv2.VideoWriter(
    'output_video2.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (original_width, original_height)
)

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

    # Loop over the detections and apply Grad-CAM++ only for cars (CAR_CLASS_ID = 6)
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        print(data)
        # class_id = int(data[5]) if len(data) == 6 else int(data[6])
        class_id=int(data[5])

        if float(confidence) < CONFIDENCE_THRESHOLD or class_id != 2:
            continue  # Skip non-car detections

        # Get bounding box coordinates
        x_min, y_min, x_max, y_max = map(int, data[:4])

        # Backpropagate the score for the detected class (car)
        model.zero_grad()
        target = detections.boxes[:, 5] == 2.0
        target = target.float().mean().to(device)
        target.backward(retain_graph=True)

        # Apply Grad-CAM++ to obtain the heatmap
        heatmap = apply_gradcam_plus()

        # Resize the heatmap to the size of the bounding box
        heatmap_resized = cv2.resize(heatmap, (x_max - x_min, y_max - y_min))

        # Convert the heatmap to an RGB image
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        # Overlay the heatmap on the original frame within the bounding box
        overlay = frame.copy()
        overlay[y_min:y_max, x_min:x_max] = heatmap_resized * 0.4 + frame[y_min:y_max, x_min:x_max]

        # Display the frame with the Grad-CAM++ overlay
        cv2.imshow("Grad-CAM++ (Cars Only)", overlay)
        output_video.write(frame)
        
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()

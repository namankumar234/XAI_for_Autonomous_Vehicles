# XAI_for_Autonomous_Vehicles

This repository contains the implementation of an Explainable AI (XAI) model for autonomous vehicle detection using camera inputs. The project leverages techniques like **Grad-CAM**, **LRP (Layer-wise Relevance Propagation)**, and **Grad-CAM++** to provide visual explanations for the decisions made by the AI model.

## 📌 Features
- **Object Detection**: Detect vehicles in real-time using camera feeds.
- **Explainability**: Visualize the decision-making process using XAI techniques:
  - Grad-CAM
  - Grad-CAM++
  - LRP
- **Customizability**: Easily integrate with other AI models and datasets.

## 🔧 Technologies Used
- Programming Language: **Python**
- Frameworks:
  - **TensorFlow** or **PyTorch** (Specify based on your implementation)
  - **OpenCV** for image processing
- XAI Libraries:
  - **Captum** (for PyTorch users)
  - **tf-explain** (for TensorFlow users)
- Visualization: **Matplotlib**, **Seaborn**

🚀 How to Run
1. Clone the Repository
```bash
git clone https://github.com/namankumar/XAI_for_Autonomous_Vehicles.git
cd XAI_for_Autonomous_Vehicles
```

2. Install Dependencies
Ensure Python is installed. Then, run:
```bash
pip install -r requirements.txt
```

3. Run different .py files and .ipynb notebooks according to usage.

📊 Results
Heatmaps: Highlight regions in the image that the model focused on.
Metrics: Evaluate model performance using metrics like accuracy, precision, recall, and mAP.
Sample Grad-CAM Visualization:
![Grad-CAM Example]([https://your-image-hosting-link.com/image.png](https://drive.google.com/file/d/1jLZOSYfv4au_6q8YCI0iyiqhXW_EV8pt/view?usp=sharing))

📚 References
Selvaraju, R.R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks
Samek, W., et al. (2017). Explainable AI: Interpreting and Explaining Deep Learning Models
🤝 Contributing
Contributions are welcome! Please submit a pull request or open an issue for any improvements or bug fixes.




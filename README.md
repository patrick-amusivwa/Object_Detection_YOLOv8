
# 🛣️ Lane Detection Using Cascaded U-Net

<div align="center">
  <a href="https://lane-detection-cascaded-unet.streamlit.app/">
    <img src="https://img.shields.io/badge/Streamlit-Live%20App-blue?style=for-the-badge&logo=streamlit" alt="Streamlit App">
  </a>
</div>

A real-time deep learning web application for **lane detection** built with **Streamlit** and powered by a **Cascaded U-Net** architecture. This model provides **pixel-wise segmentation** to detect road lanes from uploaded images, optimized for self-driving research and vision-based transport solutions.

---

## 📌 Table of Contents
- [🧠 Abstract](#-abstract)
- [🚀 Features](#-features)
- [📂 Dataset Overview](#-dataset-overview)
- [🧪 Methodology](#-methodology)
- [📈 Performance](#-performance)
- [⚙️ Installation](#-installation)
- [📊 Evaluation](#-evaluation)
- [🛠️ Future Scope](#️-future-scope)
- [📄 License](#-license)
- [📬 Contact](#-contact)

---

## 🧠 Abstract

This project explores an advanced lane detection system using a **Cascaded U-Net** deep learning model. The system is deployed as a lightweight web app using **Streamlit**, enabling users to upload road scene images and visualize the detected lane segmentation masks instantly.

The model takes an RGB road image and outputs a **binary lane mask**, trained using supervised learning. The pipeline utilizes **TensorFlow/Keras**, with support from image processing libraries such as NumPy and Pillow.

---

## 🚀 Features

✅ Upload road scene images (`.jpg`, `.png`, `.webp`)  
✅ Real-time lane segmentation using a pretrained Cascaded U-Net  
✅ Side-by-side visualization of original vs predicted masks  
✅ Saves uploaded and predicted images for later analysis  
✅ Fully browser-accessible (no installation required)

---

## 📂 Dataset Overview

> ⚠️ This repository does not include training scripts. A pre-trained `model_for_lane.keras` file is provided.

The model was trained on a curated dataset of **road images with lane mask annotations**, ensuring robust generalization to diverse road scenarios. The architecture has been validated on both synthetic and real-world road datasets.

---

## 🧪 Methodology

```text
              +--------------------------+
              |  Road Scene Image Input  |
              +------------+-------------+
                           ↓
              +------------v-------------+
              |    Preprocessing (Resize,|
              |    Normalize, etc.)      |
              +------------+-------------+
                           ↓
              +------------v-------------+
              |    Cascaded U-Net Model  |
              +------------+-------------+
                           ↓
              +------------v-------------+
              |   Binary Lane Mask Output|
              +--------------------------+
````

* **Model:** Custom U-Net cascade with encoder-decoder depth
* **Activation:** Sigmoid (for binary segmentation)
* **Loss Function:** Binary Cross-Entropy
* **Optimizer:** Adam

---

## 📈 Performance

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 97.3% |
| Precision | 96.1% |
| Recall    | 95.8% |
| F1-Score  | 95.9% |
| IoU Score | 92.4% |

> Note: Metrics computed using test set from training phase, not computed in Streamlit app.

---

## ⚙️ Installation

### 🔧 Clone & Set Up

```bash
git clone https://github.com/yourusername/lane-detection-app.git
cd lane-detection-app
```

### 🧪 Create a Virtual Environment

```bash
python -m venv venv
# Activate it
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 📦 Install Requirements

```bash
pip install -r requirements.txt
```

> `requirements.txt` includes:
>
> ```
> numpy
> pandas
> matplotlib
> keras
> tensorflow
> Pillow
> ipython
> streamlit
> ```

### 🚀 Launch the App

```bash
streamlit run app.py
```

Then open your browser at **[http://localhost:8501](http://localhost:8501)**

---

## 📊 Evaluation

Upload any road image through the UI, and the app will display:

* 🖼️ Original image
* 🔍 Predicted lane segmentation mask

Saved image directories:

* `static/uploaded/`: Uploaded images
* `static/predicted/`: Predicted lane masks

---

## 🛠️ Future Scope

* 🔄 Integrate video input for real-time inference on dashcam feeds
* 📹 Frame-by-frame streaming lane detection
* 📉 Add training and fine-tuning scripts
* 🧪 Evaluation with benchmark datasets like TuSimple, CULane
* 🧊 Migrate to ONNX/TFLite for lightweight inference

---

## 📄 License

This project is licensed under the **[MIT License](LICENSE)**. Feel free to use, modify, and distribute with attribution.

---

## 📬 Contact

Made with 💡 by **Akshwin T**
📧 [akshwint.2003@gmail.com](mailto:akshwint.2003@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/akshwin/)
🔗 [GitHub](https://github.com/akshwin)

---

> ⭐ **If you found this useful, don’t forget to star the repo and share it with your peers!**


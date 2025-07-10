
# 🛣️ Lane Detection Using Cascaded U-Net

A deep learning web app for real-time **lane detection** using a **Cascaded U-Net architecture**, built with **Streamlit** and deployed on **Streamlit Cloud**.

🎯 Ideal for autonomous driving research, computer vision learning, and road safety applications.

---

## 🌐 Live Demo

[![Streamlit App](https://img.shields.io/badge/Visit%20App-Click%20Here-blue?style=for-the-badge)](https://lane-detection-cascaded-unet.streamlit.app/)

---

## 🚀 Features

- 📤 Upload road images (JPG/PNG)
- 🤖 Predict lane segmentation using a Cascaded U-Net model
- 🖼️ Display original and predicted lane masks side-by-side
- 📁 Saves uploaded and predicted images in separate folders
- ⚡ Clean, fast, and fully interactive UI using Streamlit
- ☁️ Fully cloud-hosted – no setup needed to test

---

## 🧠 Tech Stack

- **Frontend/Backend**: Streamlit
- **Deep Learning**: TensorFlow / Keras
- **Image Processing**: NumPy, Pillow
- **Model**: Cascaded U-Net for binary lane segmentation
- **Deployment**: Streamlit Cloud

---

## 📁 Project Structure

```

lane-detection-app/
├── app.py                    # Streamlit web app
├── model\_for\_lane.keras      # Pre-trained Cascaded U-Net model
├── static/
│   ├── uploaded/             # Folder for uploaded road images
│   └── predicted/            # Folder for predicted lane mask outputs
├── upload\_image/
│   ├── 1.jpg                 # Sample image
│   └── road.webp             # Sample image
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── LICENSE                   # License info (MIT)
├── .gitignore                # Files ignored by Git

````

---

## ⚙️ Getting Started Locally

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/lane-detection-app.git
cd lane-detection-app
````

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
# Activate it
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

Then open the app in your browser:
**[http://localhost:8501](http://localhost:8501)**

---

## 🖼️ How to Use

1. Upload a road scene image using the file uploader.
2. The model processes the image and generates a binary mask for detected lanes.
3. View both **original** and **predicted** images side-by-side.

Uploaded and output images are saved to:

* 📁 `static/uploaded/` — original images
* 📁 `static/predicted/` — predicted lane masks

---

## 📦 Requirements

Your `requirements.txt` should include:

```
streamlit
tensorflow
numpy
Pillow
matplotlib
```

Add any other libraries used for your model or preprocessing.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

Made with 💡 by **Akshwin T**
📧 [akshwint.2003@gmail.com](mailto:akshwint.2003@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/akshwin/)
🔗 [GitHub](https://github.com/akshwin)


---

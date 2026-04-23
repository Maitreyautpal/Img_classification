# 🔥 Wildlife Image Classification App

An AI-powered image classification web app built using **CNN (MobileNet)** and deployed with Streamlit.

---

## 🌍 Live Demo

👉 https://imgclassification-riqxekwnavtyucejphgvqd.streamlit.app/

---

## 🚀 Features

* 🧠 Deep Learning model (CNN - MobileNet)
* 🖼️ Upload any image for prediction
* 📊 Displays prediction confidence
* 🔝 Shows Top-3 predictions
* ⚡ Fast and lightweight model (~11MB)
* 🎨 Beautiful UI built with Streamlit

---

## 📂 Project Structure

```
Img_classification/
│── app.py
│── mobilenet_best.keras
│── class_labels.json
│── requirements.txt
```

---

## ⚙️ Installation (Run Locally)

### 1️⃣ Clone the repository

```
git clone https://github.com/Maitreyautpal/Img_classification.git
cd Img_classification
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run the app

```
streamlit run app.py
```

---

## 🧪 How to Use

1. Open the app (link above 👆)
2. Upload an image (jpg/png)
3. Wait for prediction
4. View:

   * 🐾 Predicted class
   * 📊 Confidence score
   * 🔝 Top 3 predictions

---

## 🧠 Model Details

* Architecture: CNN (MobileNet-based)
* Input Size: 224x224
* Output Classes: 17 wildlife categories
* Training: ImageDataGenerator with augmentation
* EarlyStopping used to avoid overfitting

---

## 📊 Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Streamlit

---

## 💡 Future Improvements

* 🎥 Live webcam prediction
* 📊 Graph visualization of probabilities
* 🌐 Custom domain deployment
* ⚡ Model optimization for faster inference

---

## 🙌 Author

**Maitreya Utpal, Shriyansh verdhan singh, Kishan Dubey, Utkarsh Shukla**

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!

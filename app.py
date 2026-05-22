import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import base64
from tensorflow.keras.utils import load_img,img_to_array
model=tf.keras.models.load_model('model.h5')
st.title('Face Mask Detection')
if 'page' not in st.session_state:
    st.session_state.page="Home"
#for bakcground image
def add_bg():

    with open("background.png", "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background:
            linear-gradient(rgba(0,0,0,0.65),
            rgba(0,0,0,0.65)),
            url("data:image/png;base64,{encoded}");

            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )
if st.session_state.page=="Home":
    add_bg()
    input=st.file_uploader('Upload Your file',type=['jpg','jpeg','png'])
    if(st.button('Predict')):
        if input is not None:
            img_ori=load_img(input,target_size=(200,200))
            img=img_to_array(img_ori)
            img=np.expand_dims(img,axis=0)
            img=img/255.0
            Y_pred=model.predict(img)
            confidence=Y_pred[0][0]*100
            if confidence >= 80:
                st.image(img_ori)
                st.success("No Mask")
            elif confidence <=20:
                st.image(img_ori)
                st.success("Mask")
            else:
                st.image(img_ori)
                st.error("Please Enter Valid image")
        else:
            st.error('Please upload an image')
st.sidebar.title("Face Mask Detection")
if st.sidebar.button('Home'):
    st.session_state.page="Home"
    st.rerun()
if st.sidebar.button('Project Journey'):
    st.session_state.page="About"
    st.rerun()
if st.session_state.page=="About":
    st.markdown("""

# 😷 Face Mask Detection Using CNN

## 🚀 My Journey Building This Project

This project started when I wanted to learn more about **Computer Vision** and **Deep Learning** by building something practical and useful.

So I decided to create a **Face Mask Detection System** using CNN that can detect whether a person is wearing a mask or not from an image.

---

## 📂 Dataset

I used a dataset containing two classes:

- `with_mask`
- `without_mask`

This helped me understand how image datasets work in real-world deep learning projects.

---

## 🖼 Image Preprocessing

Before training the model, I performed several preprocessing steps like:

- Resizing images
- Converting images into arrays
- Normalizing pixel values
- Expanding dimensions for CNN input

This was important because CNN models require all images in the same format and size.

---

## 🧠 Building the CNN Model

Initially, I built the model using:

- Conv2D Layers
- MaxPooling Layers
- Batch Normalization
- Flatten Layer
- Dense Layers

At first, I used a `Flatten()` layer to convert feature maps into a single vector before passing them to Dense layers.

But later, I noticed that the model was:

- Overfitting
- Giving wrong predictions
- Validation accuracy was very low

I learned that `Flatten()` was creating a huge number of parameters, making the model heavy and less efficient.

So I removed the `Flatten()` layer and replaced it with:

- `GlobalAveragePooling2D()`

because it reduces the number of parameters and helps the model generalize better on new images.

After making this change, the model became more stable and prediction performance improved.

---

## ⚡ Challenges I Faced

While building this project, I faced many issues such as:

- Wrong predictions
- Overfitting
- Validation accuracy stuck near 50%
- Confidence score problems
- Label mapping confusion
- Model predicting only one class

At one point, the model accuracy looked good on training data but failed completely on validation images.

---

## 🔥 How I Improved the Model

To improve the model performance, I:

- Fixed label handling
- Tuned the CNN architecture
- Added Dropout layers
- Removed Flatten layer
- Used GlobalAveragePooling2D
- Improved confidence threshold logic
- Used proper validation data
- Applied normalization correctly

These changes helped improve the overall prediction accuracy.

---

## 🌐 Streamlit Deployment

After training the model, I deployed it using **Streamlit** and created a simple web application where users can:

- Upload images
- Predict Mask / No Mask
- View confidence score
- Handle unknown images

---

## 🛠 Technologies Used

- Python
- TensorFlow
- Keras
- CNN
- NumPy
- Streamlit
- Matplotlib

---

## 📈 What I Learned

This project helped me understand:

- How CNNs work
- Image preprocessing
- Model training and validation
- Overfitting handling
- Streamlit deployment
- Real-world AI project workflow

Most importantly, this project taught me that building AI models is not only about writing code, but also about debugging, experimenting, and improving the model step by step.

""",unsafe_allow_html=True)
    
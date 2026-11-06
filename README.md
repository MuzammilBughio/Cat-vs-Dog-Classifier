# 🐶🐱 Dog vs Cat Classifier

A Streamlit web app that classifies an image as a dog or a cat using a
MobileNet-based transfer learning model.

## Features
- Upload your own image or pick one of 13 sample images
- Predict button with result + confidence percentage
- "About Model" panel in the header

## Project structure
```
.
├── app.py
├── cat_vs_dog_model.keras   # your trained model (add this)
├── requirements.txt
└── samples/                 # put up to 13 dog/cat images here
```

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (free)
Push the folder to GitHub and deploy on Streamlit Community Cloud.

from fastai.learner import load_learner
from fastai.vision.core import PILImage

import streamlit as st
from PIL import Image


def predict_img(img):
    """Infer using trained image model

    Pass    PILImage object
    Return  prediction[str], prediction_idx[int], probability[tensor]
    """
    if img is not None:
        return learner_inf.predict(pil_img)


"## Bird Species Classifier"
"Here's the [GitHub](https://github.com/jacKlinc/bird_classifier) repo"
"Upload a picture of a bird"
learner_inf = load_learner("models/birds_full.pkl")

# Upload
pic = st.file_uploader("Upload Files")

"Click Classify"

probs = []
pred_idx = 1
pred = "n/a"

# Display image
if pic is not None:
    img = Image.open(pic)
    st.image(img)

    # Parse image
    pil_img = PILImage.create(pic)

    # Predict category
    pred, pred_idx, probs = predict_img(pil_img)

    # Classify
    if st.button("Classify"):
        prob = round(probs[pred_idx].item(), 5)
        "Probability: ", str(prob)
        "Prediction: ", pred
        #"Prediction: ", pred if prob > 0.5 else 'Not sure'


import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the image classification model
model = load_model("keras_Model.h5", compile=False)

# Load the labels and remove newline characters
class_names = [label.strip() for label in open("labels.txt", "r").readlines()]

# Mapping of diseases to symptoms
disease_symptoms = {
    'cataract': 'Symptoms for Cataract: Blurred vision, Colors seem faded, Glare',
    'glaucoma': 'Symptoms for Glaucoma: Patchy blind spots, Tunnel vision, Severe headache',
    'diabetic retinopathy': 'Symptoms for diabetic retinopathy: Distorted vision, Straight lines appear wavy, Dark, blurry areas',
    'Normal Eye': 'No  symptoms',
    # Add more diseases and their symptoms as needed
}

# Set up the page with a colorful background
st.markdown(
    """
    <style>
        body {
            background-color: #f9fbe7; /* Light Green */
        }
        .main-container {
            background-color: #ffffff; /* White */
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .image-container {
            text-align: center;
            margin-top: 20px;
        }
        .disease-label {
            font-size: 24px;
            color: #e91e63; /* Pink */
            margin-top: 20px;
        }
        .predicted-class {
            font-size: 30px;
            color: #673ab7; /* Deep Purple */
            margin-top: 10px;
        }
        .confidence-score {
            font-size: 20px;
            color: #2196f3; /* Blue */
            margin-top: 10px;
        }
        .symptoms {
            font-size: 18px;
            color: #ff5722; /* Deep Orange */
            margin-top: 10px;
        }
        .file-uploader {
            background-color: #4caf50; /* Green */
            color: #ffffff; /* White */
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            cursor: pointer;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Eye Image Diagnosis')

# Create file upload option for image classification with colorful button
uploaded_image = st.file_uploader("Choose an image for classification", type=["jpg", "jpeg", "png"], key="file_uploader", help="Allowed file types: jpg, jpeg, png",)

# Add a markdown element with the specified class for styling
if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")

    # Resize the image to be at least 224x224 and then crop from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Create the array of the right shape to feed into the Keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict using the image classification model
    prediction = model.predict(data)
    
    index = np.argmax(prediction)
    predicted_disease = class_names[index]
    confidence_score = prediction[0][index]
    confidence_score = confidence_score * 100
    
    # Display prediction, confidence score, and symptoms with colorful elements
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="disease-label">Disease:</div>', unsafe_allow_html=True)
    st.markdown('<div class="predicted-class">{}</div>'.format(predicted_disease), unsafe_allow_html=True)
    st.markdown('<div class="confidence-score">Confidence Score: {:.2f}</div>'.format(confidence_score), unsafe_allow_html=True)
    
    # Display symptoms based on the predicted disease
    if predicted_disease[2:] in disease_symptoms:
     st.markdown('<div class="symptoms">', unsafe_allow_html=True)
     st.markdown('**Disease Name:** {}'.format(predicted_disease[2:]))
     st.markdown('**Confidence Score:** {:.2f}'.format(confidence_score))
     st.markdown('**Disease Symptoms:**')
    
    symptoms_list = disease_symptoms[predicted_disease[2:]].split(', ')
    for symptom in symptoms_list:
        st.markdown('- {}'.format(symptom))

    st.markdown('</div>', unsafe_allow_html=True)
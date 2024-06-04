import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2

st.set_page_config(
    page_title="Detection System",
    page_icon="🔍",
    initial_sidebar_state="auto"
)

class VideoTransformer(VideoTransformerBase):
    def __init__(self, model):
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = Image.fromarray(img)
        img = img.resize((128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(img)
        input_arr = np.array([input_arr])
        predictions = self.model.predict(input_arr)
        result_index = np.argmax(predictions)
        confidence = predictions[0][result_index] * 100

        class_name = ['Acne', 'Eczema', 'Melanoma', 'Normal']
        result_text = f'{class_name[result_index]}: {confidence:.2f}%'
        img = np.array(img)
        img = cv2.putText(img, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(img, format='bgr24')

# Tensorflow model prediction
def model_prediction(input_image, model):
    try:
        image = Image.open(io.BytesIO(input_image.read()))  # Read the content as bytes
        image = image.resize((128, 128))  # Ensure the image is the correct size for the model
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to batch

        # Perform the prediction
        predictions = model.predict(input_arr)
        result_index = np.argmax(predictions)
        confidence = predictions[0][result_index] * 100

        return result_index, confidence
    except Exception as e:
        st.error(f"Error in model prediction: {e}")
        return None, None

# Load the trained model
model_path = "cnn_skin_disease_model.h5"
try:
    trained_model = tf.keras.models.load_model(model_path)
    st.success(f"Model loaded successfully from {model_path}")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    trained_model = None

st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #bff2ca;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for video transformer
if 'video_transformer' not in st.session_state:
    if trained_model:
        st.session_state['video_transformer'] = VideoTransformer(trained_model)
    else:
        st.session_state['video_transformer'] = None

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("SKIN DISEASE DETECTION SYSTEM")
    st.markdown("""
    Welcome to the Skin Disease Detection System! 🔍
    
    Our mission is to help in identifying skin diseases efficiently. Scan the surface of skin, and our system will analyze it to detect any signs of diseases. Together, let's protect our skin and ensure a healthier body!

    ### How It Works
    1. **Scan:** Go to the **Disease Recognition** page and scan the surface of skin with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Skin Disease Recognition System!
    """)
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    input_method = st.selectbox("Select input method:", ["Upload Image", "Live Camera"])
    
    if input_method == "Upload Image":
        input_image = st.file_uploader("Choose an Image:", type=['jpg', 'png', 'jpeg'])
        if input_image:
            st.image(input_image, use_column_width=True)
            
            # Predicting Image
            if st.button("Predict"):
                st.write("Our Prediction")
                if trained_model:
                    result_index, confidence = model_prediction(input_image, trained_model)
                    if result_index is not None:
                        class_name = ['Acne', 'Eczema', 'Melanoma', 'Normal']
                        model_predicted = class_name[result_index]
                        st.success(f"Model is predicting it's {model_predicted} with {confidence:.2f}% confidence")
                    else:
                        st.error("Prediction failed. Please try again.")
                else:
                    st.error("Model not loaded. Please check the model file.")
    elif input_method == "Live Camera":
        if trained_model:
            if st.session_state.video_transformer:
                webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=lambda: st.session_state.video_transformer)
                if webrtc_ctx.video_transformer:
                    st.write("Using live camera input for prediction")
            else:
                st.error("Video transformer is not initialized.")
        else:
            st.error("Model not loaded. Please check the model file.")

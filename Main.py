# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
import tensorflow as tf
import numpy as np

st.set_page_config(
    page_title="Detection System",
    page_icon= "🔍",
    initial_sidebar_state = "auto"
)    

# Tensorflow model prediction
def model_prediction(input_image):
    trained_model = tf.keras.models.load_model("cnn_skin_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(input_image,target_size=(228,228))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) # To convert single image to batch
    predictions = trained_model.predict(input_arr)
    result_index = np.argmax(predictions)

    return result_index

st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #bff2ca;
    }
</style>
""", unsafe_allow_html=True)

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("SKIN DISEASE DETECTION SYSTEM")
    #image_path = "home_page.jpeg"
    #st.image(image_path,use_column_width=True)
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
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!
    """)
elif choose == "Disease Recognition":
    st.header("Disease Recognition")
    input_image = st.file_uploader("Choose an Image:",type=['jpg', 'png', 'jpeg'])
    if not input_image:
        input_image = "Images/test.jpg"
    if st.button("show image"):
        st.image(input_image, use_column_width=True)
    
    # Predicting Image
    if st.button("Predict"):
        st.write("Our Prediction")
        result_index = model_prediction(input_image)
        class_name = ['Acne', 'Actinic Keratosis', 'Eczema', 'Melanoma', 'Normal', 'Rosacea']

        model_predicted = class_name[result_index]
        st.success("Model is Predicting it's a {}".format(model_predicted))





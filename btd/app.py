import tensorflow as tf
from PIL import Image
import streamlit as st
import numpy as np 
model = tf.keras.models.load_model("braintumorcorrectmodelhope.h5") #model load

st.title("🧠 Brain Tumor Detector")
st.header("Upload An MRI image Of The Brain ")
st.caption("⚠️Please Do Not Use This For A Medical Diagnosis")

uploaded_file = st.file_uploader("Choose an MRI Image", type=["jpg", "jpeg", "png"])

def preprocessed_image(image):
    img = image.convert("L")
    img = img.resize((150, 150))
    img_array = np.array(img)/255.0
    img_array = np.stack([img_array]*3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
    

if uploaded_file is not None:
    with st.spinner("Uploading"):
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
        prediction = model.predict(preprocessed_image(image))
        confidence = float(prediction[0][0])*100

    if prediction[0][0] > 0.5:
        st.subheader("Tumor Likely")
        st.error(f"{confidence:.2f}% confidence")
        st.toast("Tumor Likely")
    else:
        st.subheader("No Tumor Detected")
        st.success(f"{100-confidence:.2f}% confidence")
        st.toast("No Tumor Detected")

with st.sidebar:
    with st.expander("Works Cited"):
        st.write('''i will fix this bit''') 
st.divider() 
col1, col2 = st.columns(2)
with col1:
    with st.expander("What Is A Tumor?"):
        st.write('''A tumor (neoplasm) is an atypical agglomeration of cells within the body. Formed by cells which lack an inhibitor for their growth, tumors can be either benign (will not metastasize to other parts of the body (non-cancerous)) or malignant (likely to metastasize (cancerous)). As the name implies, brain tumors occur near or within the brain. While the exact cause is uncertain, likely causes include genetics (such as a lack or mutation of the TP53 gene –or more commonly in brain cancer mutations of the EGFR gene (which controls cell growth)), exposure to radiation and/or carcinogens or a metastatic disease which spreads to the brain. Symptoms of the tumor vary as they are dependent on the location of said tumor. ''')

with col2:
    with st.expander("How Does This Model Work?"):
        st.write('''i will fix this bit later''')

import streamlit as st
import requests
import base64
from PIL import Image
import io
import time
import os

st.set_page_config(page_title="Multi-Model Image Generator", layout="centered")
MODAL_URL = os.getenv("MODAL_URL")

st.title("Multi-Model Image Generator")
st.markdown("Compare AI-generated images from different models side by side.")

with st.expander("How to use"):
    st.write("""
    1. Upload a guide image (JPEG/PNG)
    2. Enter a prompt describing your desired output
    3. Click Generate & Compare
    4. View images from multiple AI models
    """)

# Input Section
with st.form("generation_form"):
    uploaded_file = st.file_uploader("Upload Guide Image", type=["jpg", "jpeg", "png"])
    prompt = st.text_area("Prompt", placeholder="e.g., 'a woman in traditional Indian attire standing in a sunset field'")
    submitted = st.form_submit_button("Generate & Compare")

# Model display names mapping
MODEL_DISPLAY_NAMES = {
    "controlnet_canny": "Edge-Guided Art",
    "controlnet_depth": "3D Depth Art",
    "controlnet_pose": "Pose-Aware Art",
    "controlnet_scribble": "Sketch-to-Art",
    "controlnet_normal": "Surface-Normal Art",
    "controlnet_tile": "Tile-Based Art"
}

# Run Generation
if submitted:
    if not uploaded_file:
        st.warning("Please upload an image.")
        st.stop()
    if not prompt.strip():
        st.warning("Please enter a prompt.")
        st.stop()

    with st.spinner("Generating images... please wait."):
        progress_bar = st.progress(10)
        status_text = st.empty()
        
        try:
            image_bytes = uploaded_file.read()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            progress_bar.progress(30)
            status_text.text("Calling generation API...")

            start_time = time.time()
            response = requests.post(
                MODAL_URL,
                json={"prompt": prompt, "input_image_b64": image_b64},
                headers={"Content-Type": "application/json"},
                timeout=180
            )

            progress_bar.progress(80)
            status_text.text("Processing results...")

            if response.status_code == 200:
                result = response.json()
                if "images" in result:
                    elapsed = time.time() - start_time
                    st.success(f"Generation completed in {elapsed:.1f} seconds.")
                    st.image(Image.open(io.BytesIO(image_bytes)), caption="Your Uploaded Image", use_container_width=True)

                    st.markdown("### Model Outputs")
                    images = result["images"]
                    
                    # Create columns based on number of returned images
                    cols = st.columns(len(images))
                    
                    for i, (model_key, img_b64) in enumerate(images.items()):
                        with cols[i]:
                            image_data = base64.b64decode(img_b64)
                            image = Image.open(io.BytesIO(image_data))
                            
                            # Use the display name from our mapping
                            display_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
                            st.image(image, caption=display_name, use_container_width=True)

                            # Download Button
                            buffered = io.BytesIO()
                            image.save(buffered, format="PNG")
                            st.download_button(
                                label="Download",
                                data=buffered.getvalue(),
                                file_name=f"{display_name.replace(' ', '_')}.png",
                                mime="image/png"
                            )
                else:
                    st.error("Unexpected response format from API.")
            else:
                st.error(f"API error: {response.status_code} - {response.text}")

            progress_bar.progress(100)
        except requests.exceptions.Timeout:
            st.error("Request timed out.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            status_text.empty()
            progress_bar.empty()

# Sidebar: API Status
with st.sidebar:
    st.header("API Status")
    if st.button("Check API Health"):
        try:
            health_url = MODAL_URL.replace("/compare", "/health")
            res = requests.get(health_url, timeout=10)
            if res.status_code == 200:
                st.success("API is healthy")
                st.json(res.json())
            else:
                st.error(f"Status {res.status_code}")
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")

    st.markdown("---")
    st.markdown("### Models Used")
    st.markdown("""
    - **Edge-Guided Art** – Creates art based on image edges and contours  
    - **3D Depth Art** – Generates with understanding of depth and spatial relationships  
    - **Pose-Aware Art** – Maintains human poses and body positions accurately  
    - **Sketch-to-Art** – Transforms rough sketches into refined artwork  
    - **Surface-Normal Art** – Generates with accurate surface lighting and textures  
    - **controlnet_tile** – Generates art with a focus on tile-based patterns and structures
    """)
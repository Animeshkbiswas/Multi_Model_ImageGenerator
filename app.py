import streamlit as st
import requests
import base64
from PIL import Image
import io
import time
import os

# Configure page
st.set_page_config(
    page_title="SDXL Multi-Model Generator", 
    layout="centered"
)

# Get API URL from environment
MODAL_URL = os.getenv("MODAL_URL", "https://your-modal-app.modal.run/compare")

# Custom CSS for better styling
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stDownloadButton>button {
        background-color: #2196F3;
        color: white;
        border-radius: 5px;
        padding: 0.3rem 0.7rem;
        font-size: 0.9rem;
    }
    .model-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #f9f9f9;
    }
    </style>
""", unsafe_allow_html=True)

# App header
st.title("SDXL Multi-Model Generator")
st.markdown("Generate and compare images using different SDXL ControlNet models")

# Model display configuration
MODEL_CONFIG = {
    "base": {
        "name": "SDXL Base",
        "description": "Standard SDXL generation with refiner"
    },
    "depth": {
        "name": "Depth Map",
        "description": "3D depth-aware generation"
    },
    "pose": {
        "name": "Pose Detection",
        "description": "Pose-preserving generation"
    },
    "canny": {
        "name": "Edge Detection",
        "description": "Edge-guided generation"
    }
}

# Sidebar with info and controls
with st.sidebar:
    st.header("Settings")
    with st.expander("Generation Parameters"):
        strength = st.slider("Image strength", 0.1, 1.0, 0.7, 0.05)
        guidance_scale = st.slider("Guidance scale", 1.0, 15.0, 8.0, 0.5)
        steps = st.slider("Inference steps", 10, 50, 30, 5)
    
    st.markdown("---")
    st.header("About")
    st.markdown("""
    This app uses Stable Diffusion XL with ControlNet to generate images using different conditioning methods.
    
    **Available Models:**
    - SDXL Base - Standard generation
    - Depth Map - Uses 3D depth information
    - Pose Detection - Preserves human poses
    - Edge Detection - Guided by image edges
    """)
    
    if st.button("Check API Health"):
        try:
            health_url = MODAL_URL.replace("/compare", "/health")
            res = requests.get(health_url, timeout=10)
            if res.status_code == 200:
                st.success("API is healthy")
            else:
                st.error(f"API Error: {res.status_code}")
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")

# Main content area
with st.expander("How to use", expanded=True):
    st.write("""
    1. Upload an image (optional but recommended for ControlNet models)
    2. Enter your prompt describing what you want to generate
    3. Adjust generation parameters if needed
    4. Click "Generate Images" button
    5. Compare results from different models
    """)

# Input form
with st.form("generation_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload guide image (optional)", 
            type=["jpg", "jpeg", "png"],
            help="Image to use as reference for generation"
        )
    
    with col2:
        prompt = st.text_area(
            "Prompt", 
            placeholder="A futuristic cityscape at sunset...",
            height=100,
            help="Describe what you want to generate"
        )
    
    submitted = st.form_submit_button(
        "Generate Images", 
        type="primary",
        use_container_width=True
    )

# Handle generation
if submitted:
    if not prompt.strip():
        st.warning("Please enter a prompt")
        st.stop()
    
    with st.spinner("Generating images... This may take 1-2 minutes"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Prepare request data
            request_data = {
                "prompt": prompt,
                "strength": strength,
                "guidance_scale": guidance_scale,
                "steps": steps
            }
            
            # Add image if uploaded
            if uploaded_file:
                image_bytes = uploaded_file.read()
                request_data["input_image_b64"] = base64.b64encode(image_bytes).decode("utf-8")
                progress_bar.progress(20)
                status_text.text("Processing uploaded image...")
            
            # Call API
            progress_bar.progress(40)
            status_text.text("Calling generation API...")
            
            start_time = time.time()
            response = requests.post(
                MODAL_URL,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=3000
            )
            
            # Process results
            progress_bar.progress(80)
            status_text.text("Processing results...")
            
            if response.status_code == 200:
                results = response.json()
                elapsed = time.time() - start_time
                
                st.success(f"Generation completed in {elapsed:.1f} seconds")
                
                # Show original if uploaded
                if uploaded_file:
                    st.subheader("Original Image")
                    st.image(
                        Image.open(io.BytesIO(image_bytes)), 
                        caption="Your uploaded image",
                        use_container_width=True
                    )
                
                # Display results in columns
                st.subheader("Generated Images")
                cols = st.columns(len(results))
                
                for i, (model_key, img_b64) in enumerate(results.items()):
                    if model_key == "error":
                        continue
                        
                    config = MODEL_CONFIG.get(model_key, {"name": model_key})
                    
                    with cols[i]:
                        with st.container():
                            st.markdown(f"<div class='model-card'>", unsafe_allow_html=True)
                            
                            if img_b64 == "generation_failed":
                                st.warning(config['name'])
                                st.error("Generation failed")
                            else:
                                st.markdown(f"**{config['name']}**")
                                st.caption(config.get("description", ""))
                                
                                image_data = base64.b64decode(img_b64)
                                image = Image.open(io.BytesIO(image_data))
                                st.image(image, use_container_width=True)
                                
                                # Download button
                                buffered = io.BytesIO()
                                image.save(buffered, format="PNG")
                                st.download_button(
                                    label="Download",
                                    data=buffered.getvalue(),
                                    file_name=f"{config['name'].lower().replace(' ', '_')}.png",
                                    mime="image/png",
                                    use_container_width=True
                                )
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                
                progress_bar.progress(100)
                status_text.empty()
                
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                progress_bar.empty()
                status_text.empty()
                
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()
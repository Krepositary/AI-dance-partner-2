import streamlit as st
from colab import AIDancePartner
import tempfile
import os
import time
import cv2
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="AI Dance Partner",
    page_icon="💃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
def local_css():
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            background-color: #FF4B4B;
            color: white;
            border-radius: 20px;
            padding: 0.5rem 2rem;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #FF6B6B;
            border-color: #FF4B4B;
        }
        .upload-text {
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 1rem;
        }
        .title-container {
            background: linear-gradient(90deg, #FF4B4B, #FF8C8C);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        .info-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

def get_video_preview(video_path):
    """Generate a preview frame from the video"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    return None

def main():
    local_css()
    
    # Title section with gradient background
    st.markdown("""
        <div class="title-container">
            <h1>🕺 AI Dance Partner 💃</h1>
            <p style="font-size: 1.2rem;">Transform your solo dance into a dynamic duet!</p>
        </div>
    """, unsafe_allow_html=True)

    # Create two columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<p class="upload-text">Upload your dance video and watch the magic happen!</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=['mp4', 'avi', 'mov'])

        if uploaded_file is not None:
            # Create a temporary file for the uploaded video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_file.read())
                temp_input_path = tfile.name

            # Show video preview
            st.markdown("### 📽️ Preview")
            preview_image = get_video_preview(temp_input_path)
            if preview_image:
                st.image(preview_image, use_container_width=True, caption="Video Preview")
            
            # Add video player for original
            st.markdown("### 🎥 Original Video")
            st.video(temp_input_path)

    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### How it works")
        st.markdown("""
        1. Upload your solo dance video
        2. Choose your preferred dance style
        3. Watch as AI creates your perfect dance partner!
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            st.markdown("### 🎭 Choose Your Dance Partner Style")
            style = st.select_slider(
                "",
                options=["Sync Partner", "Creative Partner"],
                value="Sync Partner"
            )
            
            if style == "Sync Partner":
                st.info("💫 Sync Partner will mirror your movements in perfect harmony.")
            else:
                st.info("🎨 Creative Partner will add its own artistic flair to your dance.")
            
            if st.button("Generate Dance Partner 🎬"):
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    steps = [
                        "Analyzing dance moves...",
                        "Detecting pose landmarks...",
                        "Generating partner movements...",
                        "Creating final video..."
                    ]
                    
                    for i, step in enumerate(steps):
                        status_text.text(step)
                        progress_bar.progress((i + 1) * 25)
                        time.sleep(0.5)
                    
                    # Process video
                    dance_partner = AIDancePartner()
                    output_path = dance_partner.process_video(temp_input_path, mode=style)
                    
                    # Update progress
                    progress_bar.progress(100)
                    status_text.text("Done! 🎉")
                    
                    # Display result
                    st.markdown("### 🎥 Your Dance Duet")
                    st.video(output_path)
                    
                    # Download button
                    with open(output_path, 'rb') as file:
                        st.download_button(
                            label="Download Video 📥",
                            data=file,
                            file_name="ai_dance_partner.mp4",
                            mime="video/mp4"
                        )
                    
                    # Cleanup temporary files
                    os.unlink(temp_input_path)
                    os.unlink(output_path)
                    
                except Exception as e:
                    st.error(f"Oops! Something went wrong: {str(e)}")
                    if os.path.exists(temp_input_path):
                        os.unlink(temp_input_path)

if __name__ == "__main__":
    main()

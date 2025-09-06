import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw
import io
import base64
import json
from colorization_model import ColorizationModel
import os
os.environ["STREAMLIT_SUPPRESS_WATCHER_WARNINGS"] = "true"

# Page configuration
st.set_page_config(
    page_title="AI Image Colorization",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86C1;
        margin-bottom: 30px;
        padding: 20px;
        background: linear-gradient(90deg, #E3F2FD, #F3E5F5);
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .feature-card {
        background: #F8F9FA;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #2E86C1;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .step-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }

    .success-box {
        background: #D4EDD4;
        border: 2px solid #28a745;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }

    .warning-box {
        background: #FFF3CD;
        border: 2px solid #FFC107;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }

    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }

    .metric-container {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def apply_sobel(image_data, kernel_size=3):
    """Apply Sobel edge detection from raw image bytes"""
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Failed to decode image from URL.")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=kernel_size)

    sobel_combined = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_normalized = np.uint8(255 * sobel_combined / np.max(sobel_combined))
    sobel_normalized = cv2.bitwise_not(sobel_normalized)

    return image, sobel_normalized


def create_brush_canvas(original_image, edge_image):
    """Create interactive brush canvas with automatic prediction"""
    original_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    edge_pil = Image.fromarray(edge_image)

    original_buffer = io.BytesIO()
    original_pil.save(original_buffer, format='PNG')
    original_b64 = base64.b64encode(original_buffer.getvalue()).decode()

    edge_buffer = io.BytesIO()
    edge_pil.save(edge_buffer, format='PNG')
    edge_b64 = base64.b64encode(edge_buffer.getvalue()).decode()

    height, width = edge_image.shape
    max_display_width = 600
    scaling_factor = max_display_width / width
    width = int(width * scaling_factor)
    height = int(height * scaling_factor)

    html_code = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                background: #f8f9fa;
                margin: 0;
                padding: 20px;
            }}
            .canvas-container {{
                background: white;
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                max-width: 700px;
                margin: 0 auto;
            }}
            canvas {{
                border: 3px solid #667eea;
                border-radius: 10px;
                cursor: crosshair;
                touch-action: none;
                display: block;
                margin: 0 auto;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            }}
            .controls {{
                margin: 20px 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 15px;
            }}
            .brush-size {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .brush-size input[type="range"] {{
                width: 120px;
                height: 6px;
                border-radius: 3px;
                background: rgba(255, 255, 255, 0.3);
                outline: none;
            }}
            .brush-size input[type="range"]::-webkit-slider-thumb {{
                appearance: none;
                width: 18px;
                height: 18px;
                border-radius: 50%;
                background: white;
                cursor: pointer;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
            }}
            .button-group {{
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
            }}
            button {{
                padding: 10px 20px;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-weight: bold;
                transition: all 0.3s ease;
                font-size: 14px;
            }}
            button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            }}
            .clear-btn {{
                background: #e74c3c;
                color: white;
            }}
            .clear-btn:hover {{
                background: #c0392b;
            }}
            .predict-btn {{
                background: #27ae60;
                color: white;
            }}
            .predict-btn:hover {{
                background: #229954;
            }}
            .status {{
                margin: 15px 0;
                padding: 10px;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }}
            .status.success {{
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }}
            .status.info {{
                background: #cce7ff;
                color: #004085;
                border: 1px solid #b8daff;
            }}
            .brush-preview {{
                width: 30px;
                height: 30px;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.3);
                border: 2px solid white;
                display: inline-block;
                margin-left: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="canvas-container">
            <div class="controls">
                <div class="brush-size">
                    <label>🖌️ Brush Size: </label>
                    <input type="range" id="brushSize" min="5" max="50" value="20">
                    <span id="brushSizeValue">20</span>px
                    <div class="brush-preview" id="brushPreview"></div>
                </div>
                <div class="button-group">
                    <button onclick="clearCanvas()" class="clear-btn">🗑️ Clear</button>
                    <button onclick="saveCanvas()" class="predict-btn">💾 Save Canvas</button>
                </div>
            </div>

            <div id="status" class="status info">
                👆 Click and drag to add color hints, then save the canvas and upload it below for AI colorization!
            </div>

            <canvas id="displayCanvas" width="{width}" height="{height}" draggable="false"></canvas>
            <canvas id="revealCanvas" width="{width}" height="{height}" style="display:none"></canvas>
        </div>

        <script>
            const displayCanvas = document.getElementById('displayCanvas');
            const revealCanvas = document.getElementById('revealCanvas');
            const displayCtx = displayCanvas.getContext('2d');
            const revealCtx = revealCanvas.getContext('2d');

            const brushSizeSlider = document.getElementById('brushSize');
            const brushSizeValue = document.getElementById('brushSizeValue');
            const brushPreview = document.getElementById('brushPreview');
            const status = document.getElementById('status');

            let isDrawing = false;
            let brushSize = 20;
            let imageLoaded = false;
            let hasUserInput = false;

            const edgeImage = new Image();
            const originalImage = new Image();

            edgeImage.src = 'data:image/png;base64,{edge_b64}';
            originalImage.src = 'data:image/png;base64,{original_b64}';

            edgeImage.onload = () => {{
                displayCtx.drawImage(edgeImage, 0, 0, displayCanvas.width, displayCanvas.height);
            }};

            originalImage.onload = () => {{
                revealCtx.fillStyle = '#808080';
                revealCtx.fillRect(0, 0, revealCanvas.width, revealCanvas.height);
                imageLoaded = true;
            }};

            function updateBrushPreview() {{
                brushPreview.style.width = Math.min(brushSize, 30) + 'px';
                brushPreview.style.height = Math.min(brushSize, 30) + 'px';
            }}

            brushSizeSlider.addEventListener('input', function() {{
                brushSize = this.value;
                brushSizeValue.textContent = this.value;
                updateBrushPreview();
            }});
            
            function saveCanvas() {{
                const link = document.createElement('a');
                link.download = 'scribble_mask.png';
                link.href = revealCanvas.toDataURL();
                link.click();
            }}

            updateBrushPreview();

            displayCanvas.addEventListener('mousedown', startDrawing);
            displayCanvas.addEventListener('mousemove', draw);
            displayCanvas.addEventListener('mouseup', stopDrawing);
            displayCanvas.addEventListener('mouseout', stopDrawing);
            displayCanvas.addEventListener('touchstart', handleTouch);
            displayCanvas.addEventListener('touchmove', handleTouch);
            displayCanvas.addEventListener('touchend', stopDrawing);

            function startDrawing(e) {{
                if (!imageLoaded) return;
                isDrawing = true;
                hasUserInput = true;
                draw(e);
            }}

            function draw(e) {{
                if (!isDrawing) return;

                const rect = displayCanvas.getBoundingClientRect();
                const x = (e.clientX || e.touches?.[0]?.clientX) - rect.left;
                const y = (e.clientY || e.touches?.[0]?.clientY) - rect.top;

                displayCtx.save();
                displayCtx.beginPath();
                displayCtx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
                displayCtx.clip();
                displayCtx.drawImage(originalImage, 0, 0, displayCanvas.width, displayCanvas.height);
                displayCtx.restore();

                revealCtx.save();
                revealCtx.beginPath();
                revealCtx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
                revealCtx.clip();
                revealCtx.drawImage(originalImage, 0, 0, revealCanvas.width, revealCanvas.height);
                revealCtx.restore();
            }}

            function stopDrawing() {{
                isDrawing = false;
            }}

            function handleTouch(e) {{
                e.preventDefault();
                const touch = e.touches[0];
                const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 
                                                e.type === 'touchmove' ? 'mousemove' : 'mouseup', {{
                    clientX: touch.clientX,
                    clientY: touch.clientY
                }});
                displayCanvas.dispatchEvent(mouseEvent);
            }}

            function clearCanvas() {{
                displayCtx.drawImage(edgeImage, 0, 0, displayCanvas.width, displayCanvas.height);
                revealCtx.fillStyle = '#808080';
                revealCtx.fillRect(0, 0, revealCanvas.width, revealCanvas.height);
                hasUserInput = false;
                status.textContent = '👆 Click and drag to add color hints, then click "Predict Colors" to see the AI colorization!';
                status.className = 'status info';
            }}

            function predictColorization() {{
                if (!hasUserInput) {{
                    status.textContent = '⚠️ Please add some color hints first by brushing on the image!';
                    status.className = 'status info';
                    return;
                }}

                status.textContent = '🔄 Processing colorization... Please wait!';
                status.className = 'status info';

                // Convert canvas to base64 and trigger Streamlit update
                const scribbleData = revealCanvas.toDataURL('image/png');

                // Send data to Streamlit via custom event
                window.parent.postMessage({{
                    type: 'scribble_data',
                    data: scribbleData
                }}, '*');

                status.textContent = '✅ Colorization request sent! Check the result below.';
                status.className = 'status success';
            }}
        </script>
    </body>
    </html>
    """
    return html_code, height + 300


# Initialize session state
if 'scribble_data' not in st.session_state:
    st.session_state.scribble_data = None
if 'model' not in st.session_state:
    st.session_state.model = None

# Main UI
st.markdown(
    '<div class="main-header"><h1>🎨 AI Image Colorization Studio</h1><p>Transform your sketches into beautiful colored images with AI!</p></div>',
    unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🚀 Quick Start Guide")
    st.markdown('<div class="step-card">1️⃣ Enter image URL</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-card">2️⃣ Adjust edge detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-card">3️⃣ Brush color hints</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-card">4️⃣ Save & upload canvas</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-card">5️⃣ Get AI colorization</div>', unsafe_allow_html=True)


# Main content



st.markdown("### 📸 Image Input")
image_url = st.text_input("🔗 Enter Image URL:", placeholder="https://example.com/image.jpg")

    # if st.button("📂 Or upload from computer"):
    #     st.info("📝 Note: Direct file upload coming soon! For now, please use image URLs.")



if image_url:
    try:
        with st.spinner("🔄 Processing image..."):
            img_data = requests.get(image_url).content

            # Edge detection settings
            st.markdown("### 🎯 Edge Detection Settings")
            strength = st.slider("Sobel kernel size", min_value=1, max_value=7, value=3, step=2,
                                 help="Higher values create thicker edges")

            original_image, sobel_result = apply_sobel(img_data, strength)
            original_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            orig_width, orig_height = original_pil.size

        # Display results
        st.markdown("### 🖼️ Image Processing Results")
        col1, col2 = st.columns(2)

        with col1:
            # st.markdown('<div class="feature-card"><h4>📷 Original Image</h4></div>', unsafe_allow_html=True)
            st.image(original_image, caption="Original", channels="BGR", use_container_width=True)

        with col2:
            # st.markdown('<div class="feature-card"><h4>🎨 Edge Detection</h4></div>', unsafe_allow_html=True)
            st.image(sobel_result, caption=f"Sobel Edges (Kernel={strength})", channels="GRAY", use_container_width=True)

        # Interactive canvas
        st.markdown("### 🖌️ Interactive Colorization Canvas")


        html_canvas, dynamic_height = create_brush_canvas(original_image, sobel_result)

        # Handle canvas interaction
        st.components.v1.html(html_canvas, height=dynamic_height)

        # Initialize model
        if st.session_state.model is None:
            with st.spinner("🤖 Loading AI model..."):
                try:
                    st.session_state.model = ColorizationModel("weights/imroved_unet_epoch_050.pt")
                    st.success("✅ Model loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading model: {e}")
                    st.info("🔧 Please check if the model file exists in the correct path.")

        # Alternative approach: Manual upload with better UX
        st.markdown("### 🎨 Generate AI Colorization")

        # col1, col2 = st.columns([2, 1])
        # with col1:
        st.markdown(
            '<div class="warning-box">📝 <strong>How to use:</strong><br>1. Brush color hints on the canvas above<br>2. Right-click the canvas → "Save image as..." → save as PNG<br>3. Upload the saved image below<br>4. Click "Generate Colorization"</div>',
            unsafe_allow_html=True)

        # with col2:
        #     if st.button("🔄 Auto-capture Canvas", help="Automatically capture the current canvas state"):
        #         st.info("🚧 Auto-capture feature coming soon! Please use manual upload for now.")

        # File upload for scribble layer
        uploaded_scribble = st.file_uploader(
            "📎 Upload your scribble layer (PNG file)",
            type=['png'],
            help="Save the canvas as PNG and upload it here"
        )

        if uploaded_scribble is not None and st.session_state.model is not None:
            try:
                # Process uploaded scribble
                scribble_pil = Image.open(uploaded_scribble).convert("RGB")

                # Generate prediction
                st.markdown("### 🎨 AI Colorization Result")

                with st.spinner("🔮 AI is working its magic..."):
                    sketch_pil = Image.fromarray(sobel_result).convert("RGB")
                    output_image = st.session_state.model.predict(sketch_pil, scribble_pil)
                    output_image = output_image.resize((orig_width, orig_height))

                # Display results
                col1, col2, col3 = st.columns(3)

                with col1:
                    # st.markdown('<div class="feature-card"><h4>🖌️ Your Color Hints</h4></div>', unsafe_allow_html=True)
                    st.image(scribble_pil, caption="Scribble Layer", use_container_width=True)

                with col2:
                    # st.markdown('<div class="feature-card"><h4>📏 Edge Detection</h4></div>', unsafe_allow_html=True)
                    st.image(sobel_result, caption="Sketch Input", use_container_width=True)

                with col3:
                    # st.markdown('<div class="feature-card"><h4>🌈 AI Result</h4></div>', unsafe_allow_html=True)
                    st.image(output_image, caption="Predicted Colorization", use_container_width=True)

                # Download section
                st.markdown("### 💾 Download Results")
                col1, col2 = st.columns(2)

                with col1:
                    # Download colorized image
                    buffer = io.BytesIO()
                    output_image.save(buffer, format='PNG')
                    st.download_button(
                        label="📥 Download Colorized Image",
                        data=buffer.getvalue(),
                        file_name="colorized_image.png",
                        mime="image/png",
                        use_container_width=True
                    )

                with col2:
                    # Download original for comparison
                    orig_buffer = io.BytesIO()
                    original_pil.save(orig_buffer, format='PNG')
                    st.download_button(
                        label="📥 Download Original",
                        data=orig_buffer.getvalue(),
                        file_name="original_image.png",
                        mime="image/png",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"❌ Error processing scribble: {e}")
                st.info("🔧 Please make sure you uploaded a valid PNG file from the canvas.")

        elif uploaded_scribble is not None:
            st.warning("⚠️ Please wait for the AI model to load before uploading scribble.")

        else:
            st.info(
                "🖌️ Draw some color hints on the canvas above, then save and upload the image to see AI colorization!")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        st.markdown(
            '<div class="warning-box">⚠️ <strong>Troubleshooting:</strong><br>• Make sure the URL points to a valid image file<br>• Check your internet connection<br>• Try a different image URL</div>',
            unsafe_allow_html=True)

# Footer

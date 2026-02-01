import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils import *

# --------------------------
# Custom CSS for styling
# --------------------------
st.markdown("""
    <style>
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Background gradient */
    .stApp {
        background: linear-gradient(
            -45deg,
            #ffe5ec,
            #fff7e6,
            #e6f7ff,
            #e6ffe6
        );
        background-size: 400% 400%;
        animation: gradientBG 10s ease infinite;
        font-family: 'Segoe UI', sans-serif;

        /* Flex to center content */
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        height: 100vh;  /* full viewport height */
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Title style */
    .dashboard-title {
        text-align: center;
        font-size: 38px;
        font-weight: 700;
        color: #6a1b9a;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
        margin: 0;
        padding: 0 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Centered Title
st.markdown('<div class="dashboard-title"> Interactive Image Processing Dashboard</div>', unsafe_allow_html=True)


# --------------------------
# Image Upload
# --------------------------
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    st.image(image, caption="Uploaded Image", width=300)  # Fixed width

    # --------------------------
    # Tabs for better layout
    # --------------------------
    tabs = st.tabs(["Transform", "Drawing & ROI", "Arithmetic & Bitwise", "Filters & Thresholds", "Advanced"])

    # --------------------------
    # Transform Tab
    # --------------------------
    with tabs[0]:
        st.subheader(" Transformations")
        if st.checkbox("Resize"):
            width = st.number_input("Width", value=image_np.shape[1])
            height = st.number_input("Height", value=image_np.shape[0])
            image_np = resize_image(image_np, width, height)

        if st.checkbox("Rotate"):
            angle = st.slider("Angle", -360, 360, 0)
            image_np = rotate_image(image_np, angle)

        if st.checkbox("Flip"):
            flip_code = st.selectbox("Flip Type", ["Horizontal", "Vertical", "Both"])
            image_np = flip_image(image_np, flip_code)

        if st.checkbox("Color Conversion"):
            color_space = st.selectbox("Color Space", ["GRAY", "HSV", "LAB"])
            image_np = convert_color(image_np, color_space)

        if st.checkbox("Extract RGB values"):
            x = st.number_input("X Coordinate", value=0, min_value=0, max_value=image_np.shape[1]-1)
            y = st.number_input("Y Coordinate", value=0, min_value=0, max_value=image_np.shape[0]-1)
            r, g, b = get_rgb(image_np, x, y)
            st.write(f"RGB at ({x}, {y}): R={r}, G={g}, B={b}")

    # --------------------------
    # Drawing & ROI Tab
    # --------------------------
    with tabs[1]:
        st.subheader("✏️ Drawing & Region of Interest")
        if st.checkbox("Draw Shapes"):
            draw_type = st.selectbox("Shape", ["Line", "Rectangle", "Circle", "Text"])
            image_np = draw_shapes(image_np, draw_type, st.sidebar)

        if st.checkbox("ROI"):
            x = st.slider("X", 0, image_np.shape[1], 0)
            y = st.slider("Y", 0, image_np.shape[0], 0)
            w = st.slider("Width", 1, image_np.shape[1], 100)
            h = st.slider("Height", 1, image_np.shape[0], 100)
            roi = image_np[y:y+h, x:x+w]
            st.image(roi, caption="Region of Interest", width=400)

    # --------------------------
    # Arithmetic & Bitwise Tab
    # --------------------------
    with tabs[2]:
        st.subheader(" & Bitwise Operations")
        if st.checkbox("Addition / Subtraction"):
            uploaded_file2 = st.file_uploader("Upload second image", type=["png","jpg"], key="second")
            if uploaded_file2:
                image2 = np.array(Image.open(uploaded_file2))
                operation = st.selectbox("Operation", ["Add", "Subtract"])
                image_np = add_subtract_images(image_np, image2, operation)

        if st.checkbox("Bitwise Operations"):
            uploaded_file2 = st.file_uploader("Upload mask image", type=["png","jpg"], key="mask")
            if uploaded_file2:
                mask = np.array(Image.open(uploaded_file2))
                op = st.selectbox("Bitwise Op", ["AND", "OR", "XOR", "NOT"])
                image_np = bitwise_op(image_np, mask, op)

    # --------------------------
    # Filters & Thresholds Tab
    # --------------------------
    with tabs[3]:
        st.subheader(" Filters & Thresholds")
        if st.checkbox("Edge Detection"):
            method = st.selectbox("Method", ["Canny", "Sobel", "Laplacian"])
            image_np = edge_detection(image_np, method)

        if st.checkbox("Thresholding"):
            th_type = st.selectbox("Type", ["Simple", "Adaptive", "Otsu"])
            image_np = thresholding(image_np, th_type)

        if st.checkbox("Blur Image"):
             ksize = st.slider("Kernel Size", 1, 21, 5)
             method = st.selectbox("Blur Method", ["Gaussian", "Median", "Average", "Bilateral"])
             image_np = blur_image(image_np, ksize, method)


        if st.checkbox("Morphological Operations"):
            morph_type = st.selectbox("Operation", ["Erode", "Dilate", "Open", "Close"])
            ksize = st.slider("Kernel Size", 1, 21, 3)
            image_np = morphological_op(image_np, morph_type, ksize)

        if st.checkbox("Count Objects"):
            count, img_with_contours = count_objects(image_np, draw_contours=True)
            st.image(img_with_contours, caption=f"Objects Detected: {count}", width=300)


    # --------------------------
    # Advanced Tab
    # --------------------------
    with tabs[4]:
        st.subheader(" Advanced Features")
        if st.checkbox("Feature Matching"):
            uploaded_file2 = st.file_uploader("Upload second image", type=["png","jpg"], key="feature")
            if uploaded_file2:
                image2 = np.array(Image.open(uploaded_file2))
                image_np = feature_matching(image_np, image2)

        if st.checkbox("Image Translation"):
            tx = st.number_input("Translate X", value=0)
            ty = st.number_input("Translate Y", value=0)
            image_np = translate_image(image_np, tx, ty)

    # --------------------------
    # Show Final Processed Image
    # --------------------------
    st.write("---")
    st.image(image_np, caption="Processed Image", width=300)


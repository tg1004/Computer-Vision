import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Image Filters & Transform", layout="centered")

def pil_to_cv2(img_pil:Image.Image):
    img=np.array(img_pil)
    if img.ndim==2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def cv2_to_pil(img_cv2:np.ndarray):
    if img_cv2 is None:
        return None
    if img_cv2.ndim == 2:
        return Image.fromarray(img_cv2)
    rgb =cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def encode_image_to_bytes(img_cv2: np.ndarray, fmt="PNG"):
    pil=cv2_to_pil(img_cv2)
    buf=io.BytesIO()
    pil.save(buf,format=fmt)
    return buf.getvalue()

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, k):
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), 0)

def sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def histogram_equalization(img):
    g = to_gray(img)
    return cv2.equalizeHist(g)

def sobel_edges(img):
    g = to_gray(img)
    sx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    return cv2.convertScaleAbs(mag)

def canny_edges(img, low, high):
    g = to_gray(img)
    return cv2.Canny(g, low, high)

def rotate_img(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), float(angle), 1.0)
    return cv2.warpAffine(img, M, (w, h))

def resize_img(img, new_w, new_h):
    new_w, new_h = int(new_w), int(new_h)
    if new_w <= 0 or new_h <= 0:
        return img
    return cv2.resize(img, (new_w, new_h))

def crop_img(img, x_pct, y_pct, w_pct, h_pct):
    h, w = img.shape[:2]
    x = int(w * x_pct / 100.0)
    y = int(h * y_pct / 100.0)
    cw = int(w * w_pct / 100.0)
    ch = int(h * h_pct / 100.0)
    # clamp
    x = max(0, min(x, w-1))
    y = max(0, min(y, h-1))
    cw = max(1, min(cw, w-x))
    ch = max(1, min(ch, h-y))
    return img[y:y+ch, x:x+cw]

st.title("Image Filter & Transform")
st.write("Upload an image, choose a filter or transform, preview, and download the result.")

col1,col2=st.columns([1,1])

with col1:
    uploaded = st.file_uploader("Upload an image(PNG/JPEG)", type=["png", "jpg", "jpeg"])


with col2:
    st.markdown("**Options**")
    op = st.selectbox("Choose operation", [
        "None (original)",
        "Grayscale",
        "Gaussian Blur",
        "Sharpen",
        "Histogram Equalization (grayscale)",
        "Sobel Edges",
        "Canny Edges",
        "Rotate",
        "Resize",
        "Crop"
    ])

    if op == "Gaussian Blur":
        k = st.slider("Kernel size (odd)", 1, 51, 7, step=2)
    elif op == "Canny Edges":
        low = st.slider("Canny low threshold", 0, 500, 50)
        high = st.slider("Canny high threshold", 1, 500, 150)
    elif op == "Rotate":
        angle = st.slider("Angle (degrees)", -180, 180, 0)
    elif op == "Resize":
        rw = st.number_input("New width (px)", min_value=1, value=512)
        rh = st.number_input("New height (px)", min_value=1, value=512)
    elif op == "Crop":
        x_pct = st.slider("X start (%)", 0, 100, 10)
        y_pct = st.slider("Y start (%)", 0, 100, 10)
        w_pct = st.slider("Crop width (%)", 1, 100, 80)
        h_pct = st.slider("Crop height (%)", 1, 100, 80)

if uploaded is not None:
    try:
        img_pil=Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error("Could not open image. Error: "+ str(e))
        st.stop()

    img_cv2=pil_to_cv2(img_pil)
    result = None
    if op == "None (original)":
        result = img_cv2.copy()
    elif op == "Grayscale":
        result = to_gray(img_cv2)
    elif op == "Gaussian Blur":
        result = gaussian_blur(img_cv2, k)
    elif op == "Sharpen":
        result = sharpen(img_cv2)
    elif op == "Histogram Equalization (grayscale)":
        result = histogram_equalization(img_cv2)
    elif op == "Sobel Edges":
        result = sobel_edges(img_cv2)
    elif op == "Canny Edges":
        result = canny_edges(img_cv2, low, high)
    elif op == "Rotate":
        result = rotate_img(img_cv2, angle)
    elif op == "Resize":
        result = resize_img(img_cv2, rw, rh)
    elif op == "Crop":
        result = crop_img(img_cv2, x_pct, y_pct, w_pct, h_pct)
    else:
        result = img_cv2.copy()

    orig_disp = cv2_to_pil(img_cv2)
    result_disp = cv2_to_pil(result)

    st.markdown("### Preview")
    c1,c2=st.columns(2)

    with c1:
        st.image(orig_disp, caption = "Original Image", use_column_width=True)
    with c2:
        st.image(result_disp, caption = f"Result - {op}", use_column_width=True)

    bts=encode_image_to_bytes(result, fmt="PNG")
    st.download_button(
        label="Download Image",
        data=bts,
        file_name="result.png",
        mime="image/png"
    )
# else:
#     st.info("Upload an image to get started.")
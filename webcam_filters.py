# webcam_filters.py
import cv2
import os
import numpy as np


try:
    from filters import sobel_edges, canny as canny_fn, to_gray, sharpen
except Exception:

    def to_gray(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    def sobel_edges(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.magnitude(sx, sy)
        return cv2.convertScaleAbs(mag)
    def canny_fn(img, low, high):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(g, low, high)
    def sharpen(img):
        k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, k)

# create captures folder
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "captures")
os.makedirs(OUT_DIR, exist_ok=True)

# camera open (use CAP_DSHOW on Windows if needed)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)  # fallback
if not cap.isOpened():
    raise RuntimeError("Cannot open camera. Close other apps or try different index/backends.")

window = "Webcam Filters (press q to quit)"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)

# trackbars for blur kernel and Canny thresholds
def nothing(x):
    pass

cv2.createTrackbar("Blur k", window, 3, 31, nothing)      # must be odd -> we'll force it
cv2.createTrackbar("Canny low", window, 50, 300, nothing)
cv2.createTrackbar("Canny high", window, 150, 400, nothing)

mode = "orig"
frame_counter = 0

print("Keys: o-original | g-gray | b-blur | s-sobel | c-canny | h-sharpen | p-save | q-quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Frame read failed. Retrying...")
        continue

    # read slider values
    k = cv2.getTrackbarPos("Blur k", window)
    if k % 2 == 0: k = max(1, k-1)    # enforce odd kernel
    low = cv2.getTrackbarPos("Canny low", window)
    high = cv2.getTrackbarPos("Canny high", window)
    if high <= low:
        high = low + 1

    # apply current mode
    if mode == "orig":
        out = frame
    elif mode == "g":
        out = to_gray(frame)
    elif mode == "b":
        # cv2.GaussianBlur expects (ksize,ksize)
        out = cv2.GaussianBlur(frame, (k,k), 0)
    elif mode == "s":
        out = sobel_edges(frame)
    elif mode == "c":
        out = canny_fn(frame, low, high)
    elif mode == "h":
        out = sharpen(frame)
    else:
        out = frame

    # some filters produce single-channel images; convert to BGR for display
    if out is None:
        disp = frame
    elif out.ndim == 2:
        disp = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    else:
        disp = out

    # overlay mode text and parameter info
    label = f"Mode: {mode} | Blur-k={k} | Canny={low}/{high} | Press p to save"
    cv2.putText(disp, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

    cv2.imshow(window, disp)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('o'):
        mode = "orig"
    elif key == ord('g'):
        mode = "g"
    elif key == ord('b'):
        mode = "b"
    elif key == ord('s'):
        mode = "s"
    elif key == ord('c'):
        mode = "c"
    elif key == ord('h'):
        mode = "h"
    elif key == ord('p'):
        # save current displayed frame
        fname = os.path.join(OUT_DIR, f"capture_{frame_counter:03d}.jpg")
        cv2.imwrite(fname, disp)
        print(f"[INFO] Saved {fname}")
        frame_counter += 1

cap.release()
cv2.destroyAllWindows()

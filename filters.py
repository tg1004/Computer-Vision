import cv2
import numpy as np

def to_gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def blur(img,ksize=5):
    return cv2.GaussianBlur(img,(ksize,ksize),0)

def sharpen(img):
    kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img,-1,kernel)

def histogram_eq_gray(img):
    g=to_gray(img)
    return cv2.equalizeHist(g)

def sobel_edges(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sx=cv2.Sobel(g,cv2.CV_64F,1,0,ksize=3)
    sy=cv2.Sobel(g,cv2.CV_64F,0,1,ksize=3)
    mag=cv2.magnitude(sx,sy)
    mag=cv2.convertScaleAbs(mag)
    return mag

def canny(img,low=50,high=150):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return cv2.Canny(g,low,high)
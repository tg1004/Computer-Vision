import cv2

def resize(img,w,h):
    return cv2.resize(img,(w,h))

def rotate(img,angle):
    h,w=img.shape[:2]
    M=cv2.getRotationMatrix2D((w/2,h/2),angle,1.0)
    return cv2.warpAffine(img,M,(w,h))

def crop(img,x,y,w,h):
    return img[y:y+h,x:x+w]
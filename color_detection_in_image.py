import cv2
import numpy as np

img=cv2.imread("sample_3.jpg",cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError("Image not found")



hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
print("HSV shape:", hsv.shape)
print("HSV dtype:", hsv.dtype)

lower_blue=np.array([100,150,50], dtype=np.uint8)
upper_blue=np.array([140,255,255], dtype=np.uint8)

mask=cv2.inRange(hsv,lower_blue,upper_blue)

result=cv2.bitwise_and(img,img,mask=mask)
cv2.imshow("Original",img)
cv2.imshow("Mask",mask)
cv2.imshow("Result",result)
cv2.waitKey(0)
cv2.destroyAllWindows()
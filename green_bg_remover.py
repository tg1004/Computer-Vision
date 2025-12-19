import cv2
import numpy as np

img=cv2.imread("green_bg2.jpg")
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

lower_green=np.array([35,100,100])
upper_green=np.array([85,255,255])

mask=cv2.inRange(hsv,lower_green,upper_green)

mask_inv=cv2.bitwise_not(mask)

foreground_img=cv2.bitwise_and(img,img,mask=mask_inv)
cv2.imshow("original",img)
cv2.imshow("mask",mask)
cv2.imshow("foreground",foreground_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
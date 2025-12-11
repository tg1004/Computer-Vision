import cv2
import numpy as np

from src.filters import sobel_edges,canny
from src.transforms import crop,resize,rotate

img = cv2.imread(r"E:\image-filters\data\sample_1.jpg")

#sobel vs canny
sobel_img = sobel_edges(img)
canny_img= canny(img,50,50)

sobel_bgr=cv2.cvtColor(sobel_img,cv2.COLOR_GRAY2BGR)
canny_bgr=cv2.cvtColor(canny_img,cv2.COLOR_GRAY2BGR)

combined=np.hstack([sobel_bgr,canny_bgr])

#filters
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(img,(7,7),0)
canny = cv2.Canny(img,50,150)
gray_bgr= cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
canny_bgr=cv2.cvtColor(canny,cv2.COLOR_GRAY2BGR)
top_row=np.hstack([img,gray_bgr])
bottom_row=np.hstack([blur,canny_bgr])
final=np.vstack([top_row,bottom_row])

#transform
resized_img=resize(img,256,256)
rotated_img=rotate(img,180)
cropped_img=crop(img,90,150,256,256)



cv2.imshow("Original",img)
cv2.imshow("Sobel (left) VS Canny (right)",combined)
cv2.imshow("Final Comparision",final)
cv2.imshow("Resized image",resized_img)
cv2.imshow("Rotated image",rotated_img)
cv2.imshow("Cropped image",cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindow()


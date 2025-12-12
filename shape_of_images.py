import cv2,numpy as np,os


img=cv2.imread("sample_1.jpg")

print("shape: ",img.shape)
print("dtype: ",img.dtype)
h,w=img.shape[:2]

y,x=int(h/2),int(w/2)
b,g,r=img[y,x]
print("center pixel BGR :",b,g,r)

img[10:110, 10:210]=(0,0,255)


cv2.imwrite("sample_out.jpg", img)

cv2.imshow("modified", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
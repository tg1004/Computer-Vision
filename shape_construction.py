import cv2, numpy as np

img=cv2.imread("sample_2.jpg")
out=img.copy()

cv2.rectangle(out,(10,10),(200,200),(1,255,0),3)
cv2.circle(out,(300,300),50,(255,0,0),-1)

cv2.putText(out,"Hello",(80,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

h1,w1=img.shape[:2]
out_resized=cv2.resize(out,(w1,h1))
combo=cv2.hconcat([img,out_resized])
cv2.imshow("combo",combo)
cv2.waitKey(0)
cv2.destroyAllWindows()
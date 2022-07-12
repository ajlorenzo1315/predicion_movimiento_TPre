import cv2
import numpy as np
import glob
import os 
#sorted(glob.glob('*.png'), key=os.path.getmtime)

#print(sorted(glob.glob('*.png'), key=os.path.getmtime)) 
img_array = []
for filename in sorted(glob.glob('*.png')):
    print(filename)
    img1 = cv2.imread(filename)
    #img2 = cv2.imread("../monodepth2/"+filename)
    img3 = cv2.imread("../diffnet_1024x320_ms_ttr/"+filename)
    img4 = cv2.imread("../diffnet_1024x320_ms/"+filename)

    img5 = cv2.imread("../diffnet_640x192/"+filename)
   

    #print(img.shape)
    #print(img2.shape)
    #print(img3.shape,img4.shape)
    #print(img8.shape)

    img=np.concatenate([img1,img3], 1)
    img2=np.concatenate([img4,img5], 1)

    #print(img.shape)
    #print(img2.shape)
    #print(diff2.size,imgmd2.size)
    #img2=np.concatenate([diff2, imgmd2], 1)
    img=np.concatenate([img,img2], 0)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)



out = cv2.VideoWriter('aproject2.avi',cv2.VideoWriter_fourcc(*'DIVX'),5, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

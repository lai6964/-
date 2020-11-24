import glob
import os
import cv2
count=0
for path in glob.glob('download_room/*.jpg'):
    A=cv2.imread(path)
    h, w, _ = A.shape
    # if min(w,h)<400:
    if A is None:
        print('del ',path)
        os.remove(path)
        count+=1
print(count)

# from skimage import io
# for path in glob.glob('download_room/*.jpg'):
#     try:
#         io.imread(path)
#     except Exception as e:
#         print(e)
#         print(path)

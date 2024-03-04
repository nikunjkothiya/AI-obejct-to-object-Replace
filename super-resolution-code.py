import cv2
from cv2 import dnn_superres


sr = dnn_superres.DnnSuperResImpl_create()

# read the model
path = 'model/EDSR_x2.pb'
sr.readModel(path)

# set the model and scale
sr.setModel('edsr', 2)  # model based scale like x2,x4

# load the image
image = cv2.imread('examples/final/main.png')

# upsample the image
upscaled = sr.upsample(image)

# save the upscaled image
cv2.imwrite('upscaled_test.png', upscaled)
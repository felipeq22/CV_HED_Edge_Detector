# Import Libraries

import cv2
from google.colab.patches import cv2_imshow
import numpy as np

# Argument Parser

class CropLayer(object):
    def __init__(self, params, blobs):
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        (inputShape, targetShape) = (inputs[0], inputs[1])
        print('inputShape ' + str(inputs[0]))
        print('targetShape ' + str(inputs[1]))
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        print('batchSize ' + str(inputShape[0]))
        print('numChannels ' + str(inputShape[1]))

        (H, W) = (targetShape[2], targetShape[3])
        print('H ' + str(inputs[2]))
        print('W ' + str(inputs[3]))
        print('inputs ' + str(inputs))
        print('inputsShape ' + str(inputShape))

        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)

        print('startX ' + str(self.startX))
        print('startY ' + str(self.startY))

        self.endX = self.startX + W
        self.endY = self.startY + H

        print('endX ' + str(self.endX))
        print('endY ' + str(self.endY))

        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):

        print('forward:')
        print([inputs[0][:, :, self.startY:self.endY,
                self.startX:self.endX]])

        return [inputs[0][:, :, self.startY:self.endY,
                self.startX:self.endX]]


protoPath = '/content/deploy.prototxt' #IT IS A TXT FILE THAT CONTAINS INFORMATION ABOUT THE NN
# MODEL.
modelPath = '/content/hed_pretrained_bsds.caffemodel'
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
cv2.dnn_registerLayer('Crop', CropLayer)

video = cv2.VideoCapture('/content/video1.mp4')


# Create an output video writer
output = cv2.VideoWriter('/content/output.mp4', cv2.VideoWriter_fourcc(*'MP4V'),
                         25, (width, height))

while video.isOpened():

    # Read a frame from the video
    ret, frame = video.read()

    if not ret:
        break

    print('In process')
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0,
                                 size=(width, height),
                                 mean=(104.00698793, 116.66876762,
                                       122.67891434),
                                 swapRB=False, crop=False)
    net.setInput(blob)
    out = net.forward()
    out = cv2.resize(out, (frame.shape[1], frame.shape[0]))

    # hed = cv2.resize(hed[0, 0], (width, height))
    # hed = (255 * hed).astype("uint8")

    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    out = 255 * out
    out = out.astype(np.uint8)

    cv2_imshow(out)
    output.write(out)

video.release()
output.release()
print('Finished')

import torch 

from torch.autograd import Variable # 
'''
converting tensors in torch variable 
torch variable contains both tensor and gradient 
'''

import cv2 

from data import BaseTransform, VOC_CLASSES as labelmap 
'''
BaseTransform: input images compatible with neural network
VOC_CLASSES : Contains Dictionary
'''
from ssd import build_ssd  # architecture 
'''
constructor of ssd nueral network
'''
import imageio

dev = torch.cuda.is_available()
def detect_(frame,net,transform):
    #frame: image on which detection is applied
    #net : ssd nueral network
    #transform: transform image to appropriate format

    height,width = frame.shape[:2] # excluding channel

    frame_t= transform(frame)[0]

    #converting into torch tensor 
    x = (torch.from_numpy(frame_t)).permute(2,0,1) 
    #permute(2,0,1) rgb to grb 

    #adding extra dimension for batches
    x=x.unsqueeze(0) # batch to zero index

    #converting into  torch variable
    if dev:
	    x = Variable(x).cuda()
    else:
	    x=Variable(x)

    #FEEDING IN SSD NEURAL NETWORK
    y = net(x)

    #Values of object
    detections = y.data #tensor
    
    #normalising val in 0 and 1
    scale = torch.Tensor([width,height,width,height]) 

    for i in range(detections.size(1)):
        j=0
        while detections[0,i,j,0] > .6:
            pts = (detections[0,i,j,1:]*scale).numpy()
            cv2.rectangle(
                frame,
                (int(pts[0]),int(pts[1])),
                (int(pts[2]),int(pts[3])),
                (255,0,0),
                2
            )
            cv2.putText(
                frame,
                labelmap[i-1],
                (int(pts[0]),int(pts[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                2, # size 
                (255,255,255),
                2, # width 
                cv2.LINE_AA # anti-alised line 
            )
            j+=1
    return frame


#CREATING SSD NN

net = build_ssd('test') # takes two phase train and test
net.load_state_dict(torch.load('/home/dragonbreath/Zenith/Python/Projects/Object Detection/ssd300_mAP_77.43_v2.pth',map_location = lambda storage, loc: storage ))
if dev:
	net.cuda() 
#CREATING THE TRANSFORMATION

transform = BaseTransform(net.size,
    (
        104/256.0,
        117/256.0,
        123/256.0
    ))

#OBJECT DETECTION ON VIDEO
imageio.plugins.ffmpeg.download()
reader = imageio.get_reader('/home/dragonbreath/a.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('/home/dragonbreath/output21.mp4',fps=fps) # no of fps u want
print(len(reader))
for i ,frame in enumerate(reader): 
    #iterate through frame of reader
    frame = detect_(frame,net.eval(),transform)
    writer.append_data(frame)
    print(i)

writer.close()

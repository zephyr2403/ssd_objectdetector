# Single Shot MultiBox Detector
## About
Single Shot MultiBox Detector utilizes a single network to identify as well as classify the objects in an image.

**MultiBox**   
In Single Shot MultiBox Detector, MultiBox is the name of the technique for bounding box regression developed by **Szegedy et al.**
MultiBox proposes coordinate for the bounding box.

MultiBox gives two important elements:  

**Location Loss(l)**: This corresponds to the difference between the predicted bounding box and ground truth bounding box.  
**Confidence Loss(c)**: It is a measure of how sure the network is, of object belonging to a particular class.  

Multibox box Loss(m), which is how correct our overall prediction is given by: *m = c + αl*  
 
Where α is used to balance the contribution of location loss.   
Jaccard Index = Area of Overlap of Bounding box/Area of Union Of Bounding Box. MultiBox endeavors to relapse nearer to the ground truth, but detection begins from prior.

**Code in this repository contains Implementation Of SDD in Pytorch**  

## Usage:

* Activate the virtual environment.
* Download and load the weight(in object_detection.py)<I'll create a link for weight>
`net.load_state_dict(torch.load('/home/dragonbreath/Zenith/Python/Projects/Object Detection/ssd300_mAP_77.43_v2.pth',map_location = lambda storage, loc: storage ))`
* Load the video File(in object_detection.py):  
 `reader = imageio.get_reader('/home/dragonbreath/a.mp4')`     
* Run object_detection.py via `python object_detection.py`  
* It will process all the frames in the video and generate the video output of detected objects.  

**Code supports both GPU(if your have cuda enabled) and CPU**
## Output
<p align="center">
  <img alt="Loading Ouput" width="800" height="500" src="output/output.gif">
</p>


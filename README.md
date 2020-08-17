# MRI tumor detection with MASK R-CNN
  
In the field of medicine, medical image analysis and processing play a vital role, especially in Non-invasive treatment and clinical study. Medical imaging techniques and analysis tools help medical practitioners and radiologists to correctly diagnose the disease. Medical Image Processing has emerged as one of the most important tools to identify and diagnose various anomalies. Imaging enables doctors to visualize and analyze the MR images for finding the abnormalities in internal structures. An important factor in the diagnosis includes the medical image data obtained from various biomedical devices which use different imaging techniques like X-rays, CT scans, MRI, mammogram, etc.
we are going to build a Mask R-CNN model capable of detecting tumours from MRI scans of the brain images.   
#Mask R-CNN architectures provide a flexible and efficient framework for parallel evaluation of region proposal (attention), object detection (classification), and instance segmentation. Preconfigured bounding boxes at various shapes and resolutions are tested for the presence of a potential abnormality. It is a challenging computer vision task that requires both successful object localization in order to locate and draw a bounding box around each object in an image, and object classification to predict the correct class of object that was localized.

## Acknowledgment
This repo borrows tons of code from  
[matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)  

## dataset BRATS 2019 ##
BraTS has always been focusing on the evaluation of state-of-the-art methods for the segmentation of brain tumors in multimodal magnetic resonance imaging (MRI) scans. BraTS 2019 utilizes multi-institutional pre-operative MRI scans and focuses on the segmentation of intrinsically heterogeneous (in appearance, shape, and histology) brain tumors, namely gliomas. Furthemore, to pinpoint the clinical relevance of this segmentation task, BraTS’19 also focuses on the prediction of patient overall survival, via integrative analyses of radiomic features and machine learning algorithms. Finally, BraTS'19 intends to experimentally evaluate the uncertainty in tumor segmentations.
'Download and unzip BraTS data from [braTS2019](https://www.med.upenn.edu/cbica/brats2019.html)'
# BRATS 2019 despite potential benefits for research prioritisation and policy setting. 

Steps applied to perform the work:
 
* Clone the repository
First, we will clone the mask rcnn repository which has the architecture for Mask R-CNN. Use the following command to clone the repository:

git clone https://github.com/matterport/Mask_RCNN.git.
* Install the dependencies
Here is a list of all the dependencies for Mask R-CNN:

numpy
scipy
Pillow
cython
matplotlib
scikit-image
tensorflow>=1.3.0
keras>=2.0.8
opencv-python
h5py
imgaug
IPython

* Download the pre-trained weights (trained on MS COCO)
Next, we need to download the pretrained weights. You can use this link(https://github.com/matterport/Mask_RCNN/releases) to download the pre-trained weights. These weights are obtained from a model that was trained on the MS COCO dataset. Once you have downloaded the weights, paste this file in the samples folder of the Mask_RCNN repository that we cloned in step 1. 

*Load Model and Make Prediction
First, the model must be defined via an instance MaskRCNN class.

This class requires a configuration object as a parameter. The configuration object defines how the model might be used during training or inference.

In this case, the configuration will only specify the number of images per batch, which will be one, and the number of classes to predict.

We can now define the MaskRCNN instance.

We will define the model as type “inference” indicating that we are interested in making predictions and not training. We must also specify a directory where any log messages could be written, which in this case will be the current working directory.













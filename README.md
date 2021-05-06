# Single Shot Detector Overview
In this project, we re-implemented a basic approach to Single Shot Detector and Object Production Problem. This repo includes some fundamental files that we need to concern: 
- **train.py** for training SSD model from scractch
- **eval.py** for evaluating pretrained SSD model from a saved checkpoint.
- **detect.py** for object detection demo. 

Due to lack of time, as well as limited in hardware requirement, we simple re-used the pretrained SSD model, which performs quite well in the end. 

# About the Dataset
**Single Shot Detector** use the **PASCAL VOC DATASET**, which is spliited into the following sub-datasets:
- [2007 _trainval_](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) (460MB)

- [2012 _trainval_](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) (2GB)

- [2007 _test_](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) (451MB) 

The pre-trained model we used was trained on both [2007 _trainval_](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) and [2012 _trainval_](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) datasets, and was tested on [2007 _test_](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) dataset

In order to run this code, you should download the data above and put them in the below order: 

data

    |
    |___trainval 
    
    |       |  
    |       |___VOC2007  
    |       |  
    |       |___VOC2012
    |
    |___test
    
            | 
            |___VOC2007
        

# Implement 
Since the code is to build a model from scratch, we can download this pretrained model [here](https://drive.google.com/open?id=1bvJfF6r_zYl2xZEpYXxgb7jLQHFZ01Qe).

Note that this checkpoint should be [loaded directly with PyTorch](https://pytorch.org/docs/stable/torch.html?#torch.load) for evaluation or inference – see below.

Run the **eval.py** file to evaluate the pretrained model.


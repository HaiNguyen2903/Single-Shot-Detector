# Single-Shot-Detector
This is a short README to show how to run this repo after cloning. 
First of all, **Single Shot Detector** use the **PASCAL VOC DATASET**, which is spliited into the following sub-datasets:
- [2007 _trainval_](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) (460MB)

- [2012 _trainval_](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) (2GB)

- [2007 _test_](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) (451MB)

In order to run this code, you should download the data above and put them in the below order:  

data

    |
    |___trainval 
  
            |  
            |___VOC2007  
            |  
            |___VOC2012
    |
    |___test

            | 
            |___VOC2007
        

Since the code is to build a model from scratch, we can download this pretrained model [here](https://drive.google.com/open?id=1bvJfF6r_zYl2xZEpYXxgb7jLQHFZ01Qe).

Note that this checkpoint should be [loaded directly with PyTorch](https://pytorch.org/docs/stable/torch.html?#torch.load) for evaluation or inference – see below.

Run the **eval.py** file to evaluate the pretrained model.

Due to lack of time, this is just a quick tutorial to use this repo =='. I know it's very sketchy, but hope it's enough to have you use this repo. 

I will update this README as soon as possible so hope you enjoy this !

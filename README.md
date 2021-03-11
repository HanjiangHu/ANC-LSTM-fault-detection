# ANC-LSTM-fault-detection
<html>The source code, pretrained models and dataset are released here for our IROS 2021 submission of "Soft Manipulator Fault Detection and Identification Using ANC-based LSTM" by Haoyuan Gu<sup>&#8224</sup>, <a href="https://hanjianghu.github.io/">Hanjiang Hu<sup>&#8224</sup></a>, Hesheng Wang* and Weidong Chen.
</html>

[![Soft Manipulator Fault Detection and Identification Using ANC-based LSTM](https://res.cloudinary.com/marcomontalbano/image/upload/v1615367674/video_to_markdown/images/youtube--w3zSbYWDjms-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/w3zSbYWDjms "Soft Manipulator Fault Detection and Identification Using ANC-based LSTM")

## Get started
Install `pytorch` and `tensorboardX` first.

Clone this repo:

 `git clone https://github.com/HanjiangHu/ANC-LSTM-fault-detection.git`

## Prepare the dataset
This repo has been well organized with dataset in `dataset` folder and pretrained models in `outputs` folder, where the experimental results could be easily reproduced or extended for further research.

The training set and validation set are randomly spilt and each sequential sample is formatted in `json` with the inputs collected from the real-time system and the corresponding labels.

## Train the model
To train the ANC-LSTM model for the first time use the following command under the `root` path of the repo.

`python train.py --name ANC_LSTM`    

For the vanilla-LSTM model without ANC module for the comparison experiment, 

`python train.py --name vanilla_LSTM --att_dim 0`  

To ine-tune the pretrained model at `XXX` iteration,

`python train.py --name ANC_LSTM --continue_train --checkpoint_epoch XXX`    

For more details about the settings of training,

`python train.py -h`


## Validation and the real-time implementation

To validate the pretrained ANC-LSTM or vanilla LSTM model at `XXX` iteration on the validation set,

`python validate.py --name ANC_LSTM --checkpoint_epoch XXX`

`python validate.py --name vanilla_LSTM --checkpoint_epoch XXX --att_dim 0`

For the real-time implementation in C/C++, get the input vector from the system at the end of each control period first. Then use `python.h` to use the functions in the `validate.py` given the real-time input to infer the real-time sequential classification results with confidence.

## More
Our paper will be available soon and welcome to our [lab](http://irmv.sjtu.edu.cn/) if you are interested in conducting more research with soft manipulator.

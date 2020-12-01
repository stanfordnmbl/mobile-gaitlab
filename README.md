# Extracting gait metrics from videos using convolutional neural networks

Training and inference scripts for predicting gait parameters from video. We use OpenPose to extract trajectories of joints

![Cerebral Palsy (CP) gait post-operative](https://health-ai.s3.amazonaws.com/static/cp-gait.gif)

Implementation of algorithms for:
"Deep neural networks enable quantitative movement analysis using single-camera videos"
by Łukasz Kidziński*, Bryan Yang*, Jennifer Hicks, Apoorva Rajagopal, Scott Delp, Michael Schwartz

## Online demo

[Try an online demo at gaitlab.stanford.edu](http://gaitlab.stanford.edu/)

## Run demo locally

To test our code follow [this notebook](https://github.com/stanfordnmbl/mobile-gaitlab/blob/master/demo/demo.ipynb). To run the demo you'll need a computer with an NVIDIA GPU, [NVIDIA docker](https://github.com/NVIDIA/nvidia-docker), and Python 3.7.

## Training

To train neural networks from scratch, using our large dataset of preprocessed videos, use [training scripts from this directory](https://github.com/stanfordnmbl/mobile-gaitlab/tree/master/training). To run the training code you need a computer with a GPU and Python 3.7.

[Download the dataset used in this project](https://simtk.org/frs/?group_id=1918)

## License 

This source code is released under [Apache 2.0 License](https://github.com/stanfordnmbl/mobile-gaitlab/blob/master/LICENSE). Stanford University has a pending patent on this technology, please contact authors or [Stanford's Office of Technology Licensing](https://otl.stanford.edu/) for details if you are interested in commercial use of this technology.

Our software relies on [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) under [a custom non-commercial license](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE). Other libraries are under permissive open source licenses. For specific licenses please refer to maintainers of packages listed in our [requirements file](https://github.com/stanfordnmbl/mobile-gaitlab/blob/master/requirements.txt). If you intend to use our software it's your obligation to make sure you comply with licensing terms of all linked software.

Processed video trajectories available [here](https://simtk.org/frs/?group_id=1918) are available under [CC BY-NC 2.0](https://creativecommons.org/licenses/by-nc/2.0/) license.

[The original video file used in the demo](https://github.com/stanfordnmbl/mobile-gaitlab/blob/master/demo/in/input.mp4) is provided by courtesy of [Gillette Children's Specialty Healthcare](https://www.gillettechildrens.org/) and should not be used for anything else than testing this repository without a written permission from the hospital.

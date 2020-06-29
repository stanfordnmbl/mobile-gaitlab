# Extracting gait metrics from videos using convolutional neural networks

Training and inference scripts for predicting gait parameters from video. We use OpenPose to extract trajectories of joints

![Cerebral Palsy (CP) gait post-operative](https://health-ai.s3.amazonaws.com/static/cp-gait.gif)

Implementation of algorithms for:
"Deep neural networks enable quantitative movement analysis using single-camera videos"
by Łukasz Kidziński*, Bryan Yang*, Jennifer Hicks, Apoorva Rajagopal, Scott Delp, Michael Schwartz

## Demo

To test our code follow [this notebook](https://github.com/stanfordnmbl/mobile-gaitlab/blob/master/demo/demo.ipynb)

## Training

To train neural networks from scratch, using our large dataset of preprocessed videos, use [training scripts from this directory](https://github.com/stanfordnmbl/mobile-gaitlab/tree/master/training).

## License 

This source code is released under [Apache 2.0 License](https://github.com/stanfordnmbl/mobile-gaitlab/blob/master/LICENSE).

Processed video trajectories available [here](https://simtk.org/frs/?group_id=1918) are available under [CC BY-NC 2.0](https://creativecommons.org/licenses/by-nc/2.0/) license.

[The original video file used in the demo](https://github.com/stanfordnmbl/mobile-gaitlab/blob/master/demo/in/input.mp4) is provided by courtesy of [Gillette Children's Specialty Healthcare](https://www.gillettechildrens.org/) and should not be used for anything else than testing this repository without a written permission from the hospital.

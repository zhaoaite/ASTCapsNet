# ASTCapsNet
 This is the source code for the paper entitled "Associated Spatio-Temporal Capsule Network for Gait Recognition", IEEE Transactions on Multimedia".
Authors: Aite Zhao, Junyu Dong, Jianbo Li, Lin Qi, and Huiyu Zhou.

# Abstract
It is a challenging task to identify a person based on her/his gait patterns. State-of-the-art approaches rely on the analysis of temporal or spatial characteristics of gait, and gait recognition is usually performed on single modality data (such as images, skeleton joint coordinates, or force signals). Evidence has shown that using multi-modality data is more conducive to gait research. Therefore, we here establish an automated learning system, with an associated spatio-temporal capsule network (ASTCapsNet) trained on multi-sensor datasets, to analyze multimodal information for gait recognition. Specifically, we first design a low-level feature extractor and a high-level feature extractor for spatio-temporal feature extraction of gait with a novel recurrent memory unit and a relationship layer. Subsequently, a Bayesian model is employed for the decisionmaking of class labels. Extensive experiments on several public datasets (normal and abnormal gait) validate the effectiveness of the proposed ASTCapsNet, compared against several state-ofthe-art methods.
Index Terms—Gait recognition, associated capsules, spatiotemporal, capsule network, multi-sensor

# Datasets
The sleep dataset can be downloaded in

Sleep Bioradiolocation Database: https://www.physionet.org/content/sleepbrl/1.0.0/
PSG: https://www.physionet.org/content/sleep-accel/1.0.0/
Pressure Map Dataset: https://www.physionet.org/content/pmd/1.0.0/
Please refer to the preprocessing and other details on these three datasets.

# Requirements
python >= 3.5
numpy >= 1.18.0
scipy
tensorflow
Other dependencies can be installed using the following command:
pip install -r requirements.txt

# Usage
Pretraining process:

python MLP_Workers.py

Recognition process:

python downstream_with_crf.py

# Citation
If you use these models in your research, please cite:

@article{Zhao2021,
author = {Zhao, Aite and Dong, Junyu and Li, Jianbo and Qi, Lin and Zhou, Huiyu},
year = {2021},
month = {01},
pages = {1-14},
title = {Associated Spatio-Temporal Capsule Network for Gait Recognition}
}


# ASTCapsNet
 This is the source code for the paper entitled "Associated Spatio-Temporal Capsule Network for Gait Recognition", IEEE Transactions on Multimedia".
Authors: Aite Zhao, Junyu Dong, Jianbo Li, Lin Qi, and Huiyu Zhou.

# Abstract
It is a challenging task to identify a person based on her/his gait patterns. State-of-the-art approaches rely on the analysis of temporal or spatial characteristics of gait, and gait recognition is usually performed on single modality data (such as images, skeleton joint coordinates, or force signals). Evidence has shown that using multi-modality data is more conducive to gait research. Therefore, we here establish an automated learning system, with an associated spatio-temporal capsule network (ASTCapsNet) trained on multi-sensor datasets, to analyze multimodal information for gait recognition. Specifically, we first design a low-level feature extractor and a high-level feature extractor for spatio-temporal feature extraction of gait with a novel recurrent memory unit and a relationship layer. Subsequently, a Bayesian model is employed for the decisionmaking of class labels. Extensive experiments on several public datasets (normal and abnormal gait) validate the effectiveness of the proposed ASTCapsNet, compared against several state-ofthe-art methods.
Index Terms—Gait recognition, associated capsules, spatiotemporal, capsule network, multi-sensor

# Datasets
Three normal gait datasets: 
-CASIA Gait dataset: http://www.cbsr.ia.ac.cn/china/Gait%20Databases%20CH.asp.
-UNITO dataset: [E. Gianaria, M. Grangetto, M. Lucenteforte, and N. Balossino, Human classification using gait features. Springer, 2014]
-SDUgait dataset :  http://mla.sdu.edu.cn/info/1006/1195.htm

Two abnormal gait datasets (neurodegenerative patients):
- NDDs dataset :https://physionet.org/physiobank/database/gaitndd/
-  PD dataset: http://physionet.org/pn3/gaitpdb/


# Requirements
- python >= 3.5
- numpy >= 1.18.0
- scipy
- tensorflow

Other dependencies can be installed using the following command:
- pip install -r requirements.txt

# Usage
You need to change the class number, number of capsules in the file of capsNet.py, capsLayer.py, and utils.py shows the dataset settings.

and then run the whole model: 

python main.py




# Citation
-If you use these models in your research, please cite:

@article{Zhao2021,

author = {Zhao, Aite and Dong, Junyu and Li, Jianbo and Qi, Lin and Zhou, Huiyu},

year = {2021},

month = {01},

pages = {1-14},

journal = {IEEE Transactions on Multimedia}, 

title = {Associated Spatio-Temporal Capsule Network for Gait Recognition}

}


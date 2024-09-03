# GWNet
This article was published in the journal Computers in Industry under the title "A generalized well neural network for surface defect segmentation in Optical Communication Devices via Template Testing comparison". The uploaded code was completed by Jie Zhang.<br />
Please cite our article:<br />
@article{NIU2023103978,<br />
title = {A generalized well neural network for surface defect segmentation in Optical Communication Devices via Template-Testing comparison},<br />
journal = {Computers in Industry},<br />
volume = {151},<br />
pages = {103978},<br />
year = {2023},<br />
issn = {0166-3615},<br />
doi = {https://doi.org/10.1016/j.compind.2023.103978},<br />
url = {https://www.sciencedirect.com/science/article/pii/S0166361523001288},<br />
author = {Tongzhi Niu and Zhiyu Xie and Jie Zhang and Lixin Tang and Bin Li and Hao Wang},<br />
keywords = {Flexible manufacturing system, Surface defect segmentation, Template-Testing comparison, Attention mechanism, Siamese networks},<br />
abstract = {Surface defect detection is an important task in the field of manufacturing, and dealing with imbalanced data is a challenge that has been addressed using methods such as anomaly detection and data augmentation. However, optical devices pose a particular challenge due to their characteristics of small batches and varying types, resulting in insufficient positive sample data and difficulty in predicting the data distribution of new batches. To address this issue, we propose a neural network that learns to compare the differences between templates and testing samples, rather than directly learning the representations of the samples. By collecting templates, the model can generalize to new batches. The challenge of extracting defect features by comparison is to remove background noise, such as displacements, deformations, and texture changes. We propose a Dual-Attention Mechanism (DAM) in the stage of feature extraction, which extracts the noise-free defect features using the non-position information of self-attention. In the stage of feature fusion, we introduce a Recurrent Residual Attention Mechanism (RRAM) to generate spatial masks that shield noise and enable multi-scale feature fusion. We evaluate our method on three datasets of Optical Communication Devices (OCDs), Printed Circuit Boards (PCBs) and Motor Commutator Surface Defects (MCSD), and demonstrate that it outperforms existing state-of-the-art methods. Our work provides a promising direction for addressing the challenge of surface defect detection in OCDs and can be generalized to other flexible manufacturing system (FMS).}<br />
}<br />

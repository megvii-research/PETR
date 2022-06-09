# PETR: Position Embedding Transformation for Multi-View 3D Object Detection



## Introduction

This repository is an official implementation of [PETR: Position Embedding Transformation for Multi-View 3D Object Detection.](https://arxiv.org/abs/2203.05625)

<div align="center">
  <img src="figs/overview.png"/>
</div><br/>

**Abstract.**  In this paper, we develop position embedding transformation
(PETR) for multi-view 3D object detection. PETR encodes the position
information of 3D coordinates into image features, producing the
3D position-aware features. Object query can perceive the 3D positionaware
features and perform end-to-end object detection. PETR achieves
state-of-the-art performance (50.4% NDS and 44.1% mAP) on standard
nuScenes dataset and ranks 1st place on the leaderboard. It can
serve as a simple yet strong baseline for future research.

## News
(06/06/2022) PETRv2 explores the effectiveness of temporal modeling and high-quality BEV segmentation. [ArXiv](https://arxiv.org/abs/2206.01256)  
(10/03/2022) PETR is now on [ArXiv](https://arxiv.org/abs/2203.05625).


## Preparation
This implementation is built upon [detr3d](https://github.com/WangYueFt/detr3d) and [mmdetection3d](https://github.com/open-mmlab/mmdetection3d). 

* Environments  
Linux, Python==3.6.8, CUDA == 11.2, pytorch == 1.9.0  
Install mmcv==1.4.0 (https://github.com/open-mmlab/mmcv)  
Install mmdet==2.24.1  (https://github.com/open-mmlab/mmdetection)  
Install mmseg==0.20.2 (https://github.com/open-mmlab/mmsegmentation)  
Install mmdet3d==0.17.0  (https://github.com/open-mmlab/mmdetection3d)

* Data  & pretrained weights  
Follow the mmdet3d to process the data (https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/data_preparation.md).  
Please put the pretrained weights into ./ckpts/


## Train & inference
```bash
git clone https://github.com/megvii-research/PETR.git
```
```bash
cd PETR
```
You can train the model following:
```bash
tools/dist_train.sh projects/configs/petr/petr_r50dcn_gridmask_p4.py 8 --work-dir work_dirs/petr_r50dcn_gridmask_p4/
```
You can evaluate the model following:
```bash
tools/dist_test.sh projects/configs/petr/petr_r50dcn_gridmask_p4.py work_dirs/petr_r50dcn_gridmask_p4/latest.pth 8 --eval bbox
```
## Main Results
We provide some results on nuScenes val set with pretrained models. These model are trained without cbgs.

| Method            | mAP      | NDS     |hours    |   Download |
|--------|----------|---------|--------|-------------|
| [**PETR-r50-c5-1408x512**](projects/configs/petr/petr_r50dcn_gridmask_c5.py)   | 30.5%     | 35.0%    | -  | [model]()        |
| [**PETR-r50-p4-1408x512**](projects/configs/petr/petr_r50dcn_gridmask_p4.py) | 31.70%     | 36.7%    | -   | [model]()    
| [**PETR-vov-p4-800x320**](projects/configs/petr/petr_vovnet_gridmask_p4_800x320.py)   | 37.8%     | 42.6%    | -  | [model]()        |
| [**PETR-vov-p4-1600x640**](projects/configs/petr/petr_vovnet_gridmask_p4_1600x640.py) | 40.40%     | 45.5%    | -   | [model]()       

## Acknowledgement
Many thanks to the authors of [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) and [detr3d](https://github.com/WangYueFt/detr3d).

## Citing PETR
If you find PETR useful in your research, please consider citing: 
```bibtex   
@article{liu2022petr,
  title={Petr: Position embedding transformation for multi-view 3d object detection},
  author={Liu, Yingfei and Wang, Tiancai and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2203.05625},
  year={2022}
}
```

# PETR: Position Embedding Transformation for Multi-View 3D Object Detection
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2203.05625)
![visitors](https://visitor-badge.glitch.me/badge?page_id=megvii-research/PETR)
<!-- ## Introduction -->

This repository is an official implementation of [PETR: Position Embedding Transformation for Multi-View 3D Object Detection.](https://arxiv.org/abs/2203.05625)

<div align="center">
  <img src="figs/overview.png"/>
</div><br/>

In this paper, we develop position embedding transformation
(PETR) for multi-view 3D object detection. PETR encodes the position
information of 3D coordinates into image features, producing the
3D position-aware features. Object query can perceive the 3D position-aware features and perform end-to-end object detection. It can
serve as a simple yet strong baseline for future research.

## News
**2022.06.06** PETRv2 explores the effectiveness of temporal modeling and high-quality BEV segmentation. [ArXiv](https://arxiv.org/abs/2206.01256)  
**2022.03.10** PETR is now on [ArXiv](https://arxiv.org/abs/2203.05625).  
**2022.03.08** PETR achieves state-of-the-art performance (50.4% NDS and 44.1% mAP) on standard nuScenes dataset.

## Preparation
This implementation is built upon [detr3d](https://github.com/WangYueFt/detr3d) and [mmdetection3d](https://github.com/open-mmlab/mmdetection3d), which can be constructed as the [instructions](https://github.com/WangYueFt/detr3d/blob/main/README.md).

* Environments  
Linux, Python==3.6.8, CUDA == 11.2, pytorch == 1.9.0, mmdet3d == 0.17.1 

* Data   
Follow the mmdet3d to process the data (https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/data_preparation.md).  
* Pretrained weights   
To verify the performance on the val set, we provide the pretrained V2-99 [weights](https://drive.google.com/file/d/1ABI5BoQCkCkP4B0pO5KBJ3Ni0tei0gZi/view?usp=sharing). The V2-99 has pretrained on DDAD15M ([weights](https://tri-ml-public.s3.amazonaws.com/github/dd3d/pretrained/depth_pretrained_v99-3jlw0p36-20210423_010520-model_final-remapped.pth)) and further trained on NuScenes train set with FCOS3D. Please put the pretrained weights into ./ckpts/

* After preparation, you will be able to see the following directory structure:  
  ```
  PETR
  ├── mmdetection3d
  ├── projects
  │   ├── configs
  │   ├── mmdet3d_plugin
  ├── tools
  ├── data
  │   ├── nuscenes
  ├── ckpts
  ├── README.md
  ```

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
We provide some results on nuScenes val set with pretrained models. These model are trained on 8x 2080ti without cbgs.

| config            | mAP      | NDS     |training    |   config |   download |
|:--------:|:----------:|:---------:|:--------:|:--------:|:-------------:|
| PETR-r50-c5-1408x512   | 30.5%     | 35.0%    | 18hours  | [config](projects/configs/petr/petr_r50dcn_gridmask_c5.py)  |   [log](https://drive.google.com/file/d/1pXT6JltfMF0PAyG17zVcoXLJEYMKVWQr/view?usp=sharing) / [gdrive](https://drive.google.com/file/d/1c5rgTpHA98dFKmQ9BJN0zZbSuBFT8_Bt/view?usp=sharing)     |
| PETR-r50-p4-1408x512 | 31.70%     | 36.7%    | 21hours   | [config](projects/configs/petr/petr_r50dcn_gridmask_p4.py)   |   [log](https://drive.google.com/file/d/1Knoid2-ZiQhl1lcTt65SROTZiuvfTGT7/view?usp=sharing) / [gdrive](https://drive.google.com/file/d/1eYymeIbS0ecHhQcB8XAFazFxLPm3wIHY/view?usp=sharing)    
| PETR-vov-p4-800x320   | 37.8%     | 42.6%    | 17hours  | [config](projects/configs/petr/petr_vovnet_gridmask_p4_800x320.py)   |   [log](https://drive.google.com/file/d/1eG914jDVK3YXvbubR8VUjP2NnzYpDvHC/view?usp=sharing) / [gdrive](https://drive.google.com/file/d/1-afU8MhAf92dneOIbhoVxl_b72IAWOEJ/view?usp=sharing)        |
| PETR-vov-p4-1600x640 | 40.40%     | 45.5%    | 36hours   | [config](projects/configs/petr/petr_vovnet_gridmask_p4_1600x640.py)   |   [log](https://drive.google.com/file/d/1XfO5fb_Nd6jhQ3foBUG7WCz0SlTlBKu8/view?usp=sharing) / [gdrive](https://drive.google.com/file/d/1SV0_n0PhIraEXHJ1jIdMu3iMg9YZsm8c/view?usp=sharing)  

Note: the models and logs are also available at [Baidu Netdisk](https://pan.baidu.com/s/1ZzbYtKz4fZtjg6FyDGhvsQ) with code `iv0a`.

## Acknowledgement
Many thanks to the authors of [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) and [detr3d](https://github.com/WangYueFt/detr3d) .


## Citation
If you find this project useful for your research, please consider citing: 
```bibtex   
@article{liu2022petr,
  title={Petr: Position embedding transformation for multi-view 3d object detection},
  author={Liu, Yingfei and Wang, Tiancai and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2203.05625},
  year={2022}
}
```
```bibtex   
@article{liu2022petrv2,
  title={PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images},
  author={Liu, Yingfei and Yan, Junjie and Jia, Fan and Li, Shuailin and Gao, Qi and Wang, Tiancai and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2206.01256},
  year={2022}
}
```
## Contact
If you have any questions, feel free to open an issue or contact us at liuyingfei@megvii.com.

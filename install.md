
## Install MMCV
```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```
examples：
```bash
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```
## Install MMDetection

```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v2.24.1 
sudo pip install -r requirements/build.txt
sudo python3 setup.py develop
cd ..
```

## Install MMSegmentation.

```bash
sudo pip install mmsegmentation==0.20.2
```

## Install MMDetection3D

```bash
git clone  https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 
sudo pip install -r requirements/build.txt
sudo python3 setup.py develop
cd ..
```

## Install PETR

```bash
git clone https://github.com/megvii-research/PETR.git
cd PETR
mkdir ckpts
mkdir data
ln -s {mmdetection3d_path} ./mmdetection3d
ln -s {nuscenes_path} ./data/nuscenes
```
examples
```bash
git clone https://github.com/megvii-research/PETR.git
cd PETR
mkdir ckpts ###pretrain weights
mkdir data ###dataset
ln -s ../mmdetection3d ./mmdetection3d
ln -s /data/Dataset/nuScenes ./data/nuscenes
```





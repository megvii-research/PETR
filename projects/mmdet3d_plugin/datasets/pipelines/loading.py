# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES
from einops import rearrange
import torch
import nori2 as nori
import refile
# from skimage import io
import io
from skimage import io as skimage_io

@PIPELINES.register_module()
class LoadMultiViewImageFromNori(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self,
                 nori_lists=None,
                 to_float32=True,
                 color_type='unchanged',
                 data_prefix=None):
        self.nori_lists = nori_lists
        self.to_float32 = to_float32
        self.color_type = color_type
        self.fetcher = nori.Fetcher()
        self.data_prefix = data_prefix
        self.name2nori = self.decode_nori_list()

    def decode_nori_list(self):
        ret = {}
        for nori_list in self.nori_lists:
            with refile.smart_open(nori_list, "r") as f:
                for line in f:
                    nori_id, filename = line.strip().split()
                    ret[filename] = nori_id
        return ret

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """

        if 'nori_ids' in results.keys():
            filename = results['nori_ids']
        else:
            filename = results['img_filename']
        if self.data_prefix is not None:
            filename = [x.replace(self.data_prefix, '/data/datasets/nuScenes').replace('/data/Dataset/', '/data/datasets/')  for x in filename]
        nori_ids = [self.name2nori[name] if name in self.name2nori.keys() else name for name in filename]
        img_bytes = [self.fetcher.get(nori_id) for nori_id in nori_ids]

        # img is of shape (h, w, c, num_views)

        try:
            img = np.stack(
                [mmcv.imfrombytes(img_byte, self.color_type) for img_byte in img_bytes], axis=-1)
        except:
            img = np.stack(
                [mmcv.imfrombytes(img_byte, self.color_type)[:900, :1600, :] for img_byte in img_bytes], axis=-1)
        
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = results['img_filename']
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]

        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

@PIPELINES.register_module()
class LoadMapsFromFiles(object):
    def __init__(self,k=None):
        self.k=k
    def __call__(self,results):
        map_filename=results['map_filename']
        maps=np.load(map_filename)
        map_mask=maps['arr_0'].astype(np.float32)
        
        maps=map_mask.transpose((2,0,1))
        results['gt_map']=maps
        maps=rearrange(maps, 'c (h h1) (w w2) -> (h w) c h1 w2 ', h1=16, w2=16)
        maps=maps.reshape(256,3*256)
        results['map_shape']=maps.shape
        results['maps']=maps
        return results


@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweepsFiles(object):
    """Load multi channel images from a list of separate channel files.
    Expects results['img_filename'] to be a list of filenames.
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, 
                sweeps_num=5,
                to_float32=False, 
                file_client_args=dict(backend='disk'),
                pad_empty_sweeps=False,
                is_nori_read=True,
                sweep_range=[3,27],
                sweeps_id = None,
                color_type='unchanged',
                sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
                test_mode=True,
                prob=1.0,
                ):

        self.sweeps_num = sweeps_num    
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.sensors = sensors
        self.test_mode = test_mode
        self.is_nori_read = is_nori_read
        self.sweeps_id = sweeps_id
        self.sweep_range = sweep_range
        self.prob = prob
        if self.is_nori_read:
            self.nori_fetcher = nori.Fetcher()
        if self.sweeps_id:
            assert len(self.sweeps_id) == self.sweeps_num

    def __call__(self, results):
        """Call function to load multi-view image from files.
        Args:
            results (dict): Result dict containing multi-view image filenames.
        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.
                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        sweep_imgs_list = []
        timestamp_imgs_list = []
        imgs = results['img']
        img_timestamp = results['img_timestamp']
        lidar_timestamp = results['timestamp']
        img_timestamp = [lidar_timestamp - timestamp for timestamp in img_timestamp]
        sweep_imgs_list.extend(imgs)
        timestamp_imgs_list.extend(img_timestamp)
        nums = len(imgs)
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                sweep_imgs_list.extend(imgs)
                mean_time = (self.sweep_range[0] + self.sweep_range[1]) / 2.0 * 0.083
                timestamp_imgs_list.extend([time + mean_time for time in img_timestamp])
                # timestamp_imgs_list.extend(img_timestamp + mean_time)
                for j in range(nums):
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    results['intrinsics'].append(np.copy(results['intrinsics'][j]))
                    results['extrinsics'].append(np.copy(results['extrinsics'][j]))
        else:
            if self.sweeps_id:
                choices = self.sweeps_id
            elif len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = [int((self.sweep_range[0] + self.sweep_range[1])/2) - 1] 
            else:
                if np.random.random() < self.prob:
                    if self.sweep_range[0] < len(results['sweeps']):
                        sweep_range = list(range(self.sweep_range[0], min(self.sweep_range[1], len(results['sweeps']))))
                    else:
                        sweep_range = list(range(self.sweep_range[0], self.sweep_range[1]))
                    choices = np.random.choice(sweep_range, self.sweeps_num, replace=False)
                else:
                    choices = [int((self.sweep_range[0] + self.sweep_range[1])/2) - 1] 
                
            for idx in choices:
                sweep_idx = min(idx, len(results['sweeps']) - 1)
                sweep = results['sweeps'][sweep_idx]
                if len(sweep.keys()) < len(self.sensors):
                    sweep = results['sweeps'][sweep_idx - 1]
                results['filename'].extend([sweep[sensor]['data_path'] for sensor in self.sensors])

                if self.is_nori_read:
                    sweep_imgs = []
                    for sensor in self.sensors:
                        img_file = io.BytesIO(self.nori_fetcher.get(sweep[sensor]["nori_id"]))
                        nori_img = skimage_io.imread(img_file)
                        nori_img = nori_img[:,:,::-1]
                        sweep_imgs.append(nori_img)
                    img = np.stack(sweep_imgs, axis=-1)
                else:
                    img = np.stack(
                    [mmcv.imread(sweep[sensor]['data_path'], self.color_type) for sensor in self.sensors], axis=-1)
                
                if self.to_float32:
                    img = img.astype(np.float32)
                img = [img[..., i] for i in range(img.shape[-1])]
                sweep_imgs_list.extend(img)
                sweep_ts = [lidar_timestamp - sweep[sensor]['timestamp'] / 1e6  for sensor in self.sensors]
                timestamp_imgs_list.extend(sweep_ts)
                for sensor in self.sensors:
                    results['lidar2img'].append(sweep[sensor]['lidar2img'])
                    results['intrinsics'].append(sweep[sensor]['intrinsics'])
                    results['extrinsics'].append(sweep[sensor]['extrinsics'])
        results['img'] = sweep_imgs_list
        results['timestamp'] = timestamp_imgs_list  

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

# @PIPELINES.register_module()
# class LoadMultiViewImageFromMultiSweepsFiles(object):
#     """Load multi channel images from a list of separate channel files.
#     Expects results['img_filename'] to be a list of filenames.
#     Args:
#         to_float32 (bool): Whether to convert the img to float32.
#             Defaults to False.
#         color_type (str): Color type of the file. Defaults to 'unchanged'.
#     """

#     def __init__(self, 
#                 sweeps_num=5,
#                 to_float32=False, 
#                 file_client_args=dict(backend='disk'),
#                 pad_empty_sweeps=False,
#                 sweep_range=[3,27],
#                 sweeps_id = None,
#                 color_type='unchanged',
#                 sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
#                 test_mode=True,
#                 prob=1.0,
#                 ):

#         self.sweeps_num = sweeps_num    
#         self.to_float32 = to_float32
#         self.color_type = color_type
#         self.file_client_args = file_client_args.copy()
#         self.file_client = None
#         self.pad_empty_sweeps = pad_empty_sweeps
#         self.sensors = sensors
#         self.test_mode = test_mode
#         self.sweeps_id = sweeps_id
#         self.sweep_range = sweep_range
#         self.prob = prob
#         if self.sweeps_id:
#             assert len(self.sweeps_id) == self.sweeps_num

#     def __call__(self, results):
#         """Call function to load multi-view image from files.
#         Args:
#             results (dict): Result dict containing multi-view image filenames.
#         Returns:
#             dict: The result dict containing the multi-view image data. \
#                 Added keys and values are described below.
#                 - filename (str): Multi-view image filenames.
#                 - img (np.ndarray): Multi-view image arrays.
#                 - img_shape (tuple[int]): Shape of multi-view image arrays.
#                 - ori_shape (tuple[int]): Shape of original image arrays.
#                 - pad_shape (tuple[int]): Shape of padded image arrays.
#                 - scale_factor (float): Scale factor.
#                 - img_norm_cfg (dict): Normalization configuration of images.
#         """
#         sweep_imgs_list = []
#         timestamp_imgs_list = []
#         imgs = results['img']
#         img_timestamp = results['img_timestamp']
#         lidar_timestamp = results['timestamp']
#         img_timestamp = [lidar_timestamp - timestamp for timestamp in img_timestamp]
#         sweep_imgs_list.extend(imgs)
#         timestamp_imgs_list.extend(img_timestamp)
#         nums = len(imgs)
#         if self.pad_empty_sweeps and len(results['sweeps']) == 0:
#             for i in range(self.sweeps_num):
#                 sweep_imgs_list.extend(imgs)
#                 mean_time = (self.sweep_range[0] + self.sweep_range[1]) / 2.0 * 0.083
#                 timestamp_imgs_list.extend([time + mean_time for time in img_timestamp])
#                 for j in range(nums):
#                     results['filename'].append(results['filename'][j])
#                     results['lidar2img'].append(np.copy(results['lidar2img'][j]))
#                     results['intrinsics'].append(np.copy(results['intrinsics'][j]))
#                     results['extrinsics'].append(np.copy(results['extrinsics'][j]))
#         else:
#             if self.sweeps_id:
#                 choices = self.sweeps_id
#             elif len(results['sweeps']) <= self.sweeps_num:
#                 choices = np.arange(len(results['sweeps']))
#             elif self.test_mode:
#                 choices = [int((self.sweep_range[0] + self.sweep_range[1])/2) - 1] 
#             else:
#                 if np.random.random() < self.prob:
#                     if self.sweep_range[0] < len(results['sweeps']):
#                         sweep_range = list(range(self.sweep_range[0], min(self.sweep_range[1], len(results['sweeps']))))
#                     else:
#                         sweep_range = list(range(self.sweep_range[0], self.sweep_range[1]))
#                     choices = np.random.choice(sweep_range, self.sweeps_num, replace=False)
#                 else:
#                     choices = [int((self.sweep_range[0] + self.sweep_range[1])/2) - 1] 
                
#             for idx in choices:
#                 sweep_idx = min(idx, len(results['sweeps']) - 1)
#                 sweep = results['sweeps'][sweep_idx]
#                 if len(sweep.keys()) < len(self.sensors):
#                     sweep = results['sweeps'][sweep_idx - 1]
#                 results['filename'].extend([sweep[sensor]['data_path'] for sensor in self.sensors])

#                 img = np.stack([mmcv.imread(sweep[sensor]['data_path'], self.color_type) for sensor in self.sensors], axis=-1)
                
#                 if self.to_float32:
#                     img = img.astype(np.float32)
#                 img = [img[..., i] for i in range(img.shape[-1])]
#                 sweep_imgs_list.extend(img)
#                 sweep_ts = [lidar_timestamp - sweep[sensor]['timestamp'] / 1e6  for sensor in self.sensors]
#                 timestamp_imgs_list.extend(sweep_ts)
#                 for sensor in self.sensors:
#                     results['lidar2img'].append(sweep[sensor]['lidar2img'])
#                     results['intrinsics'].append(sweep[sensor]['intrinsics'])
#                     results['extrinsics'].append(sweep[sensor]['extrinsics'])
#         results['img'] = sweep_imgs_list
#         results['timestamp'] = timestamp_imgs_list  

#         return results

#     def __repr__(self):
#         """str: Return a string that describes the module."""
#         repr_str = self.__class__.__name__
#         repr_str += f'(to_float32={self.to_float32}, '
#         repr_str += f"color_type='{self.color_type}')"
#         return repr_str

@PIPELINES.register_module()
class LoadMapsFromFiles_flattenf200f3(object):
    def __init__(self,k=None):
        self.k=k
    def __call__(self,results):
        map_filename=results['map_filename']
        maps=np.load(map_filename)
        map_mask=maps['arr_0'].astype(np.float32)
        
        maps=map_mask.transpose((2,0,1))
        results['gt_map']=maps
        # maps=rearrange(maps, 'c (h h1) (w w2) -> (h w) c h1 w2 ', h1=16, w2=16)
        maps=maps.reshape(3,200*200)
        maps[maps>=0.5]=1
        maps[maps<0.5]=0
        maps=1-maps
        results['map_shape']=maps.shape
        results['maps']=maps
        
        return results

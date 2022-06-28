from data_converter.nuscenes_converter_seg import  create_nuscenes_infos



if __name__ == '__main__':
    # Training settings
    create_nuscenes_infos( '/data/Dataset/nuScenes/','HDmaps-nocovers')


import h5py
import random
import numpy as np

import time
import skimage.transform
import pandas as pd


class DataLoader:
    def __init__(self, name, image_size=256):
        path_dict = {}
        path_dict['kitti'] = 'kitti/data_kitti.hdf5'
        path_dict['synthia'] = 'synthia/data_synthia.hdf5'
        path_dict['chair'] = 'shapenet/data_chair.hdf5'
        path_dict['car'] = 'shapenet/data_car.hdf5'

        file_directory = '../../Multiview2Novelview/datasets/'

        assert name in path_dict.keys()
        self.name = name
        file_path = file_directory + path_dict[name]
        self.file_path = file_path
        self.image_size = image_size
        self.pose_size = 18
        self.data = None

    def get_batched_data(self, batch_size=32, single_model=True, model_name=None, verbose=False, return_info=False, is_train=True):
        pass

    def get_specific_data(self, target_data_info):
        pass


class ObjectDataLoaderNumpy(DataLoader):
    def __init__(self, name, image_size=256, train_or_test='train'):
        super().__init__(name, image_size)
        file_name = '%s_%s_%d' % (train_or_test, name, image_size)
        self.all_images = np.load('numpy_data/%s.npy' % file_name)
        self.n_elevation = 1
        self.n_azimuth = 18
        self.n_models = self.all_images.shape[0]
        self.min_elevation = 0
        self.max_elevation = 0

    def get_image_from_info(self, model_name, az, el=-1):
        return self.all_images[model_name, az]

    def get_specific_data(self, target_data_info):
        batch_size = len(target_data_info)
        input_images = np.zeros((batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
        target_images = np.zeros((batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
        input_elevations = np.zeros((batch_size,), dtype=np.float32)
        input_azimuths = np.zeros((batch_size,), dtype=np.float32)
        target_elevations = np.zeros((batch_size,), dtype=np.float32)
        target_azimuths = np.zeros((batch_size,), dtype=np.float32)

        for i in range(batch_size):
            m, ia, ie, ta, te = target_data_info[i]
            input_images[i] = self.get_image_from_info(m, ia, ie)
            target_images[i] = self.get_image_from_info(m, ta, te)
            input_elevations[i] = ie
            input_azimuths[i] = ia
            target_elevations[i] = te
            target_azimuths[i] = ta
        return input_images, target_images, (input_elevations, input_azimuths, target_elevations, target_azimuths)

    def get_batched_data(self, batch_size=32, single_model=True, model_name=None, verbose=False, return_info=False, is_train=False):
        input_random_elevations = np.random.randint(self.n_elevation, size=batch_size)
        input_random_azimuths = np.random.randint(self.n_azimuth, size=batch_size)
        target_random_elevations = np.random.randint(self.n_elevation, size=batch_size)
        target_random_azimuths = np.random.randint(self.n_azimuth, size=batch_size)

        target_model = np.random.randint(self.n_models)

        input_images = np.zeros((batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
        target_images = np.zeros((batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
        index_infos = []
        for i in range(batch_size):
            if not single_model:
                target_model = np.random.randint(self.n_models)
            input_images[i] = self.get_image_from_info(target_model, input_random_azimuths[i], input_random_elevations[i])
            target_images[i] = self.get_image_from_info(target_model, target_random_azimuths[i], target_random_elevations[i])
            index_infos.append((target_model, input_random_azimuths[i], input_random_elevations[i], target_random_azimuths[i], target_random_elevations[i]))

        if return_info:
            data_tuple = (input_images, target_images, (input_random_elevations, input_random_azimuths, target_random_elevations, target_random_azimuths))
            return data_tuple, index_infos
        else:
            return input_images, target_images, (input_random_elevations, input_random_azimuths, target_random_elevations, target_random_azimuths)

    def get_batched_data_i_j(self, source, target, model_min_index, model_max_index):
        N = model_max_index - model_min_index

        input_random_elevations = np.repeat(0, N)
        target_random_elevations = np.repeat(0, N)

        input_random_azimuths = np.repeat(source, N)
        target_random_azimuths = np.repeat(target, N)

        input_images = self.all_images[model_min_index:model_max_index, source]
        target_images = self.all_images[model_min_index:model_max_index, target]

        return input_images, target_images, (
        input_random_elevations, input_random_azimuths, target_random_elevations, target_random_azimuths)


class SceneDataLoaderNumpy(DataLoader):
    def __init__(self, name, use_pose_matrix=False, image_size=256):
        '''
        For scene data loader, it both contains train and test dataset.
        '''
        super().__init__(name, image_size)
        df = pd.read_csv("numpy_data/%s_scene_infos.csv" % self.name)
        scene_info = df.set_index('scene_id')['scene_frame_numbers'].to_dict()
        self.image_numbers_per_scene = scene_info
        self.scene_list = list(scene_info.keys())
        self.scene_number = len(self.scene_list)

        self.train_ids = {}
        self.test_ids = {}
        np.random.seed(100)
        for scene_id, scene_frame_n in self.image_numbers_per_scene.items():
            arr = np.arange(scene_frame_n)
            np.random.shuffle(arr)
            self.train_ids[scene_id] = arr[0:int(0.8*scene_frame_n)]
            self.test_ids[scene_id] = arr[int(0.8*scene_frame_n)+1:-1]
            print(scene_id, self.test_ids[scene_id][0:20])
            print("train/test number:", len(self.train_ids[scene_id]), len(self.test_ids[scene_id]))

        self.is_pose_matrix = use_pose_matrix
        self.pose_size = 6 if not use_pose_matrix else (12 if self.name == 'kitti' else 16)

        self.max_frame_difference = 10

        self.scene_offsets = {}
        offset = 0
        for scene_id in self.scene_list:
            self.scene_offsets[scene_id] = offset
            offset += self.image_numbers_per_scene[scene_id]

        self.all_images = np.load('numpy_data/%s_image.npy' % self.name)
        self.all_poses = np.load('numpy_data/%s_pose.npy' % self.name)
        self.all_pose_matrices = np.load('numpy_data/%s_pose_matrix.npy' % self.name)

    def get_image_pose(self, scene_id, frame_n):
        image = self.all_images[self.scene_offsets[scene_id] + frame_n]
        image = image.astype(np.float32)
        image = image / 255
        if self.is_pose_matrix:
            pose = self.all_pose_matrices[self.scene_offsets[scene_id] + frame_n]
        else:
            pose = self.all_poses[self.scene_offsets[scene_id] + frame_n]
        return image, pose

    def get_single_data_tuple(self, scene_id, is_train=True):
        frame_difference = np.random.randint(-self.max_frame_difference, self.max_frame_difference)
        scene_total_length = self.image_numbers_per_scene[scene_id]
        if is_train:
            input_index = random.choice(self.train_ids[scene_id])
        else:
            input_index = random.choice(self.test_ids[scene_id])
        target_index = input_index + frame_difference
        target_index = max(min(target_index, scene_total_length - 1), 0)

        input_image, input_pose = self.get_image_pose(scene_id, input_index)
        target_image, target_pose = self.get_image_pose(scene_id, target_index)

        return (input_image, input_pose, target_image, target_pose), (input_index, target_index)

    def get_batched_data(self, batch_size=32, single_model=True, model_name=None, verbose=False, return_info=False, is_train=True):
        # load new model
        input_images = np.zeros((batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
        target_images = np.zeros((batch_size, self.image_size, self.image_size, 3), dtype=np.float32)

        input_poses = np.zeros((batch_size, self.pose_size), dtype=np.float32)
        target_poses = np.zeros((batch_size, self.pose_size), dtype=np.float32)

        id_info = []

        scene_id = random.choice(self.scene_list)
        for i in range(batch_size):
            if not single_model:
                scene_id = random.choice(self.scene_list)
            single_data, index_info = self.get_single_data_tuple(scene_id, is_train=is_train)
            input_image, input_pose, target_image, target_pose = single_data
            input_index, target_index = index_info
            id_info.append((scene_id, input_index, target_index))
            input_images[i] = input_image
            target_images[i] = target_image
            input_poses[i] = input_pose
            target_poses[i] = target_pose

        if return_info:
            data_tuple = (input_images, target_images, (input_poses, target_poses))
            return data_tuple, id_info
        else:
            return input_images, target_images, (input_poses, target_poses)

    def get_specific_data(self, target_data_infos):
        n = len(target_data_infos)

        input_images = np.zeros((n, self.image_size, self.image_size, 3), dtype=np.float32)
        target_images = np.zeros((n, self.image_size, self.image_size, 3), dtype=np.float32)

        input_poses = np.zeros((n, self.pose_size), dtype=np.float32)
        target_poses = np.zeros((n, self.pose_size), dtype=np.float32)

        for i in range(n):
            data_info = target_data_infos[i]
            scene_id = data_info[0]
            input_index = data_info[1]
            target_index = data_info[2]
            input_image, input_pose = self.get_image_pose(scene_id, input_index)
            target_image, target_pose = self.get_image_pose(scene_id, target_index)
            input_images[i] = input_image
            target_images[i] = target_image
            input_poses[i] = input_pose
            target_poses[i] = target_pose

        return input_images, target_images, (input_poses, target_poses)

    def get_batched_data_i_j(self, scene_id, difference, frame_min_index, frame_max_index):
        n = frame_max_index - frame_min_index
        input_images = np.zeros((n, self.image_size, self.image_size, 3), dtype=np.float32)
        target_images = np.zeros((n, self.image_size, self.image_size, 3), dtype=np.float32)

        input_poses = np.zeros((n, self.pose_size), dtype=np.float32)
        target_poses = np.zeros((n, self.pose_size), dtype=np.float32)
        scene_total_length = self.image_numbers_per_scene[scene_id]
        for i in range(n):
            input_frame = self.test_ids[scene_id][frame_min_index + i]
            target_frame = input_frame + difference
            target_frame = max(min(target_frame, scene_total_length - 1), 0)

            input_image, input_pose = self.get_image_pose(scene_id, input_frame)
            target_image, target_pose = self.get_image_pose(scene_id, target_frame)
            input_images[i] = input_image
            target_images[i] = target_image
            input_poses[i] = input_pose
            target_poses[i] = target_pose

        return input_images, target_images, (input_poses, target_poses)


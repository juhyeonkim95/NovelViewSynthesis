# import h5py
# import random
# import numpy as np
#
# import time
# import skimage.transform
#
#
# class DataLoader:
#     def __init__(self, name, image_size=256):
#         path_dict = {}
#         path_dict['kitti'] = 'kitti/data_kitti.hdf5'
#         path_dict['synthia'] = 'synthia/data_synthia.hdf5'
#         path_dict['chair'] = 'shapenet/data_chair.hdf5'
#         path_dict['car'] = 'shapenet/data_car.hdf5'
#
#         file_directory = '../../Multiview2Novelview/datasets/'
#
#         assert name in path_dict.keys()
#         self.name = name
#         file_path = file_directory + path_dict[name]
#         self.file_path = file_path
#         self.image_size = image_size
#         self.pose_size = 18
#         self.data = None
#
#     def get_batched_data(self, batch_size=32, single_model=True, model_name=None, verbose=False):
#         pass
#
#
# class ObjectDataLoaderNumpy(DataLoader):
#     def __init__(self, name, image_size=256, train_or_test='train'):
#         super().__init__(name, image_size)
#         file_name = '%s_%s_%d' % (train_or_test, name, image_size)
#         self.all_images = np.load('numpy_data/%s.npy' % file_name)
#         self.n_elevation = 1
#         self.n_azimuth = 18
#         self.n_models = self.all_images.shape[0]
#         self.min_elevation = 0
#         self.max_elevation = 0
#
#     def get_image_from_info(self, model_name, az, el=-1):
#         return self.all_images[model_name, az]
#
#     def get_batched_data(self, batch_size=32, single_model=True, model_name=None, verbose=False):
#
#         input_random_elevations = np.random.randint(self.n_elevation, size=batch_size)
#         input_random_azimuths = np.random.randint(self.n_azimuth, size=batch_size)
#         target_random_elevations = np.random.randint(self.n_elevation, size=batch_size)
#         target_random_azimuths = np.random.randint(self.n_azimuth, size=batch_size)
#
#         target_model = np.random.randint(self.n_models)
#
#         input_images = np.zeros((batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
#         target_images = np.zeros((batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
#
#         for i in range(batch_size):
#             if not single_model:
#                 target_model = np.random.randint(self.n_models)
#             input_images[i] = self.get_image_from_info(target_model, input_random_azimuths[i],
#                                                        input_random_elevations[i])
#             target_images[i] = self.get_image_from_info(target_model, target_random_azimuths[i],
#                                                         target_random_elevations[i])
#
#         return input_images, target_images, (
#         input_random_elevations, input_random_azimuths, target_random_elevations, target_random_azimuths)
#
#     def get_batched_data_i_j(self, source, target, model_min_index, model_max_index):
#         N = model_max_index - model_min_index
#
#         input_random_elevations = np.repeat(0, N)
#         target_random_elevations = np.repeat(0, N)
#
#         input_random_azimuths = np.repeat(source, N)
#         target_random_azimuths = np.repeat(target, N)
#
#         input_images = self.all_images[model_min_index:model_max_index, source]
#         target_images = self.all_images[model_min_index:model_max_index, target]
#
#         return input_images, target_images, (
#             input_random_elevations, input_random_azimuths, target_random_elevations, target_random_azimuths)
#
#
# class SceneDataLoader(DataLoader):
#     def __init__(self, name, use_pose_matrix=False, image_size=256):
#         super().__init__(name, image_size)
#         self.image_numbers_per_scene = {}
#         self.data = h5py.File(self.file_path, 'r')
#
#         self.find_all_models(self.data)
#         self.scene_list = list(self.image_numbers_per_scene.keys())
#         self.scene_list.sort()
#         self.scene_number = len(self.scene_list)
#
#         self.use_pose_matrix = use_pose_matrix
#         self.pose_size = 6 if not self.use_pose_matrix else (12 if self.name == 'kitti' else 16)
#
#         self.max_frame_difference = 10
#
#     def export_to_npy(self):
#         total_image_length = sum(self.image_numbers_per_scene.values())
#         images = np.zeros((total_image_length, self.image_size, self.image_size, 3), dtype=np.uint8)
#         poses = np.zeros((total_image_length, 6), dtype=np.float32)
#         pose_matrices = np.zeros((total_image_length, 12 if self.name is 'kitti' else 16), dtype=np.float32)
#
#         pointer = 0
#         for scene_id in self.scene_list:
#             for frame_number in range(self.image_numbers_per_scene[scene_id]):
#                 file_name = self.scene_frame_number_to_string(scene_id, frame_number)
#                 print(file_name)
#                 data = self.data[file_name]
#                 image = np.array(data['image'], dtype=np.uint8)
#                 pose = np.array(data['pose'], dtype=np.float32)
#                 pose_matrix = np.array(data['pose_matrix'], dtype=np.float32)
#                 images[pointer] = image
#                 poses[pointer] = pose
#                 pose_matrices[pointer] = pose_matrix.flatten()
#                 pointer += 1
#         np.save("numpy_data/%s_image.npy" % self.name, images)
#         np.save("numpy_data/%s_pose.npy" % self.name, poses)
#         np.save("numpy_data/%s_pose_matrix.npy" % self.name, pose_matrices)
#
#     def find_all_models(self, data):
#         model_set = dict()
#         for k in data.keys():
#             model_name = k.split("_")[0]
#             model_set[model_name] = model_set.get(model_name, 0) + 1
#         self.image_numbers_per_scene = model_set
#
#     def get_single_image_pose_data_from_name(self, file_name):
#         data = self.data[file_name]
#         image = np.array(data['image'], dtype=np.float32)
#         if self.image_size is not 256:
#             image = skimage.transform.resize(image, (self.image_size, self.image_size))
#         image = image / 255
#         if self.use_pose_matrix:
#             pose = np.array(data['pose_matrix'], dtype=np.float32)
#         else:
#             pose = np.array(data['pose'], dtype=np.float32)
#         return image, pose
#
#     def scene_frame_number_to_string(self, scene_id, frame_n):
#         if self.name == 'synthia' and scene_id == 'SYNTHIA-SEQS-05-FALL' and frame_n == 0:
#             print("exception in synthia dataset")
#             frame_n = 1
#         frame_idx = '0' * (6 - len(str(frame_n))) + str(frame_n)
#         return '%s_%s' % (scene_id, frame_idx)
#
#     def get_image_pose(self, scene_id, frame_n):
#         file_name = self.scene_frame_number_to_string(scene_id, frame_n)
#         return self.get_single_image_pose_data_from_name(file_name)
#
#     def get_single_data_tuple(self, scene_id):
#         frame_difference = np.random.randint(-self.max_frame_difference, self.max_frame_difference)
#         scene_total_length = self.image_numbers_per_scene[scene_id]
#         input_index = np.random.randint(scene_total_length)
#         target_index = input_index + frame_difference
#         target_index = max(min(target_index, scene_total_length - 1), 0)
#         # print("Scene info", scene_id, input_index, frame_difference)
#
#         input_image, input_pose = self.get_image_pose(scene_id, input_index)
#         target_image, target_pose = self.get_image_pose(scene_id, target_index)
#
#         return input_image, input_pose, target_image, target_pose
#
#     # def get_single_data_tuple(self, scene_id):
#     #     frame_difference = np.random.randint(-self.max_frame_difference, self.max_frame_difference)
#     #     scene_total_length = self.image_numbers_per_scene[scene_id]
#     #     input_index = np.random.randint(scene_total_length)
#     #     target_index = input_index + frame_difference
#     #     target_index = max(min(target_index, scene_total_length - 1), 0)
#     #
#     #     input_name = self.scene_frame_number_to_string(scene_id, input_index)
#     #     target_name = self.scene_frame_number_to_string(scene_id, target_index)
#     #
#     #     input_image, input_pose = self.get_single_image_pose_data_from_name(input_name)
#     #     target_image, target_pose = self.get_single_image_pose_data_from_name(target_name)
#     #
#     #     return input_image, input_pose, target_image, target_pose
#
#     def get_batched_data(self, batch_size=32, single_model=True, model_name=None, verbose=False):
#         start = time.time()
#
#         # load new model
#         input_images = np.zeros((batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
#         target_images = np.zeros((batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
#
#         input_poses = np.zeros((batch_size, self.pose_size), dtype=np.float32)
#         target_poses = np.zeros((batch_size, self.pose_size), dtype=np.float32)
#
#         scene_id = random.choice(self.scene_list)
#         for i in range(batch_size):
#             if not single_model:
#                 scene_id = random.choice(self.scene_list)
#             input_image, input_pose, target_image, target_pose = self.get_single_data_tuple(scene_id)
#
#             input_images[i] = input_image
#             target_images[i] = target_image
#             input_poses[i] = input_pose
#             target_poses[i] = target_pose
#
#         end = time.time()
#         # print(end - start, 'seconds to load images')
#         return input_images, target_images, (input_poses, target_poses)
#
#     def get_specific_data(self, target_data_infos):
#         n = len(target_data_infos)
#
#         input_images = np.zeros((n, self.image_size, self.image_size, 3), dtype=np.float32)
#         target_images = np.zeros((n, self.image_size, self.image_size, 3), dtype=np.float32)
#
#         input_poses = np.zeros((n, self.pose_size), dtype=np.float32)
#         target_poses = np.zeros((n, self.pose_size), dtype=np.float32)
#
#         for i in range(n):
#             data_info = target_data_infos[i]
#             scene_id, input_index, target_index = data_info
#             input_image, input_pose = self.get_image_pose(scene_id, input_index)
#             target_image, target_pose = self.get_image_pose(scene_id, target_index)
#             input_images[i] = input_image
#             target_images[i] = target_image
#             input_poses[i] = input_pose
#             target_poses[i] = target_pose
#
#         return input_images, target_images, (input_poses, target_poses)
#
#
# class SceneDataLoaderNumpy(SceneDataLoader):
#     def __init__(self, name, use_pose_matrix=False, image_size=256):
#         super().__init__(name, use_pose_matrix, image_size)
#         self.scene_offsets = {}
#         offset = 0
#         for scene_id in self.scene_list:
#             self.scene_offsets[scene_id] = offset
#             offset += 1
#         self.all_images = np.load('numpy_data/%s_image.npy' % self.name)
#         self.all_poses = np.load('numpy_data/%s_pose.npy' % self.name)
#         self.all_pose_matrices = np.load('numpy_data/%s_pose_matrix.npy' % self.name)
#
#     def get_image_pose(self, scene_id, frame_n):
#         image = self.all_images[self.scene_offsets[scene_id] + frame_n]
#         image = image.astype(np.float32)
#         image = image / 255
#         pose = self.all_poses[self.scene_offsets[scene_id] + frame_n]
#         return image, pose
#

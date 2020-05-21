import h5py
import random
import numpy as np

import time
import skimage.transform


class DataContainer:
    def __init__(self, name, model_limit=-1, data_load_strategy='load_all_first', train_or_test='train', eager_load=True, data_per_model=2048, image_size=256):
        path_dict = {}
        path_dict['kitti'] = 'kitti/data_kitti.hdf5'
        path_dict['synthia'] = 'synthia/data_synthia.hdf5'
        path_dict['chair'] = 'shapenet/data_chair.hdf5'
        path_dict['car'] = 'shapenet/data_car.hdf5'

        file_directory = '../Multiview2Novelview/datasets/'

        assert name in path_dict.keys()
        self.name = name
        file_path = file_directory + path_dict[name]
        self.file_path = file_path

        self.data = None
        if isinstance(model_limit, int):
            self.model_limit = (0, model_limit)
        else:
            self.model_limit = model_limit
        self.model_list = []
        self.all_model_list = []
        self.all_images_for_all_model = None

        if data_load_strategy is 'load_all_model_at_once':
            if image_size == 64:
                n = '%s_%s.npy' % (train_or_test, name)
            else:
                n = '%s_%s_%d.npy' % (train_or_test, name, image_size)
            self.all_images_for_all_model = np.load(n)
            self.model_list = [i for i in range(len(self.all_images_for_all_model))]
            print(len(self.model_list), "Sized")
        else:
            if eager_load:
                self.load_h5_data()
                self.find_all_models(self.data, self.model_limit)
            else:
                with h5py.File(self.file_path, 'r') as f:
                    self.find_all_models(f, self.model_limit)

        self.n_elevation = 1
        self.n_azimuth = 18
        self.min_elevation = 0
        self.max_elevation = 0

        self.image_caches = {}

        self.data_load_strategy = data_load_strategy
        self.use_cache = (data_load_strategy == 'use_cache')
        self.load_all_first = (data_load_strategy == 'load_all_first')

        self.MAX_MODEL_TO_LOAD_AT_ONCE = 100

        self.all_images_for_single_model = None
        self.image_size = image_size

        if self.load_all_first:
            self.all_images_for_single_model = np.zeros((3, 18, image_size, image_size, 3), dtype=np.float32)

        self.current_model = None
        self.data_per_model_counter = data_per_model
        self.data_per_model = data_per_model
        self.verbose = False

    def save_images(self):
        from PIL import Image
        import os
        folder = "%s_example_images" % self.name
        if not os.path.exists(folder):
            os.mkdir(folder)

        for i in self.model_list:
            image = self.all_images_for_all_model[i][2]
            formatted = (image * 255).astype('uint8')
            print(formatted.shape)
            im = Image.fromarray(formatted)
            im.save("%s/%d.png" % (folder, i))

    @staticmethod
    def square(x):
        return x*x

    def for_lambda(self, i):
        return self.get_image_from_info(self.model_list[i], 0, 0)

    def load_using_multiprocessing(self, number=10):
        start_time = time.time()
        from multiprocessing import Pool
        images = np.zeros((number, self.image_size, self.image_size, 3), dtype=np.float32)
        with Pool(5) as p:
            #self.get_image_from_info(self.model_list[i], 0, 0)
            x = p.map(self.for_lambda, [k for k in range(number)])
        elapsed_time = (time.time() - start_time)
        print(elapsed_time)

    def load_h5_data(self):
        self.data = h5py.File(self.file_path, 'r')
        print("H5 data Loaded")

    def export_all(self, name, export_range=None):
        s, e = export_range
        n = e-s
        export_data = np.zeros((n, self.n_azimuth, self.image_size, self.image_size, 3), dtype=np.float32)
        start_time = time.time()
        self.all_images_for_single_model = np.zeros((3, 18, self.image_size, self.image_size, 3), dtype=np.float32)
        if export_range is not None:
            for i in range(s, e, 1):
                print(i)
                model_name = self.all_model_list[i]
                el = 0
                for az in range(18):
                    file_name = "%s_%d_%d" % (model_name, az * 2, el * 10)
                    self.all_images_for_single_model[el][az] = self.get_single_image_from_path(file_name)
                export_data[i-s] = self.all_images_for_single_model[0]
        else:
            for i, model_name in enumerate(self.model_list):
                el = 0
                for az in range(18):
                    file_name = "%s_%d_%d" % (model_name, az * 2, el * 10)
                    self.all_images_for_single_model[el][az] = self.get_single_image_from_path(file_name)
                export_data[i] = self.all_images_for_single_model[0]

        elapsed_time = (time.time() - start_time)
        np.save(name, export_data)

    def find_all_models(self, data, model_number_limit):
        model_set = set()
        for k in data.keys():
            model_name = k.split("_")[0]
            model_set.add(model_name)
        self.model_list = list(model_set)
        self.model_list.sort()
        start, end = model_number_limit
        end = len(self.model_list) if end == -1 else end
        end = min(end, len(self.model_list))
        self.all_model_list = [i for i in self.model_list]
        self.model_list = self.model_list[start:end]
        #if model_number_limit != - 1:
        #    self.model_list = self.model_list[0:min(model_number_limit, len(self.model_list))]
        print("Total %d models, %d for train" % (len(model_set), len(self.model_list)))

    def get_image_from_info(self, model_name, az, el=-1):
        if self.data_load_strategy=='load_all_model_at_once':
            return self.all_images_for_all_model[model_name, az]

        file_name = "%s_%d_%d" % (model_name, az * 2, el * 10)
        if self.use_cache:
            image = self.get_image_cached(file_name=file_name) #self.data['%s/image' % file_name]
        else:
            image = self.get_single_image_from_path(file_name)

        return image

    def get_single_image_from_path(self, file_name):
        image = np.array(self.data['%s/image' % file_name], dtype=np.float32)
        image = skimage.transform.resize(image, (self.image_size, self.image_size))
        image = image / 255
        return image

    def load_all_images_of_single_model(self, model_name=None):
        start_time = time.time()
        if model_name is None:
            model_name = random.choice(self.model_list)
        for el in range(3):
            for az in range(18):
                file_name = "%s_%d_%d" % (model_name, az * 2, el * 10)
                self.all_images_for_single_model[el][az] = self.get_single_image_from_path(file_name)

        elapsed_time = (time.time() - start_time)

        if self.verbose:
            print("read all files took %.4f seconds" % (elapsed_time))

    def load_all_images_of_all_model(self):
        start_time = time.time()
        for i, model_name in enumerate(self.model_list):
            self.all_images_for_single_model = np.zeros((3, 18, self.image_size, self.image_size, 3), dtype=np.float32)
            self.load_all_images_of_single_model(model_name)
            self.all_images_for_all_model[i] = self.all_images_for_single_model
        elapsed_time = (time.time() - start_time)
        print("Size :", self.all_images_for_all_model.nbytes)
        print("read all files took %.4f seconds" % (elapsed_time))

    def get_image_cached(self, file_name):
        if file_name not in self.image_caches:
            self.image_caches[file_name] = self.get_single_image_from_path(file_name)
        return self.image_caches[file_name]

    def clear_image_cache(self):
        self.image_caches.clear()

    def get_consecutive_data(self):
        model_name = random.choice(self.model_list)
        self.load_all_images_of_single_model(model_name)

        input_random_elevations = np.repeat(np.random.randint(3), 18)
        input_random_azimuths = np.zeros(18)
        target_random_elevations = np.copy(input_random_elevations)
        target_random_azimuths = np.arange(18)

    def get_batched_data_i_j(self, source, target, model_min_index, model_max_index):
        N = model_max_index - model_min_index

        input_random_elevations = np.repeat(0, N)
        target_random_elevations = np.repeat(0, N)

        input_random_azimuths = np.repeat(source, N)
        target_random_azimuths = np.repeat(target, N)

        input_images = self.all_images_for_all_model[model_min_index:model_max_index, source]  # self.all_images_for_single_model[input_random_elevations, input_random_azimuths]
        target_images = self.all_images_for_all_model[model_min_index:model_max_index, target]  # self.all_images_for_single_model[target_random_elevations, target_random_azimuths]

        return input_images, target_images, (
        input_random_elevations, input_random_azimuths, target_random_elevations, target_random_azimuths)


    def get_batched_data_single(self, start_angle=-1, model_name=None):
        if model_name == None:
            model_name = random.choice(self.model_list)

        #self.load_all_images_of_single_model(self.current_model)
        if start_angle == -1:
            start_angle = np.random.randint(18)
        input_random_elevations = np.repeat(np.random.randint(3), 18)
        input_random_azimuths = np.repeat(start_angle, 18)
        target_random_elevations = np.copy(input_random_elevations)
        target_random_azimuths = np.arange(18) + input_random_azimuths
        target_random_azimuths = np.mod(target_random_azimuths, 18)

        input_images = self.all_images_for_all_model[model_name][input_random_azimuths]#self.all_images_for_single_model[input_random_elevations, input_random_azimuths]
        target_images = self.all_images_for_all_model[model_name][target_random_azimuths]#self.all_images_for_single_model[target_random_elevations, target_random_azimuths]

        # input_elevations_one_hot = np.eye(3, dtype=np.float32)[input_random_elevations]
        # input_azimuths_one_hot = np.eye(18, dtype=np.float32)[input_random_azimuths]
        # target_elevations_one_hot = np.eye(3, dtype=np.float32)[target_random_elevations]
        # target_azimuths_one_hot = np.eye(18, dtype=np.float32)[target_random_azimuths]
        # input_poses = np.concatenate(
        #     (input_elevations_one_hot, input_azimuths_one_hot, target_elevations_one_hot, target_azimuths_one_hot),
        #     axis=1)

        # print("copy all files took %.4f seconds" % (elapsed_time))

        return input_images, target_images, (input_random_elevations, input_random_azimuths, target_random_elevations, target_random_azimuths)

    def get_batched_data_labeled(self, batch_size=32, label_type='azimuth'):
        input_random_azimuths = np.random.randint(self.n_azimuth, size=batch_size)

        model_names = np.random.randint(len(self.model_list), size=batch_size)
        input_images = np.zeros((batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
        for i in range(batch_size):
            input_images[i] = self.get_image_from_info(model_names[i], input_random_azimuths[i])

        if label_type == 'azimuth':
            azimuth_one_hot = np.eye(self.n_azimuth, dtype=np.float32)[input_random_azimuths]
            return input_images, azimuth_one_hot
        else:
            azimuth_one_hot = np.eye(self.n_azimuth, dtype=np.float32)[input_random_azimuths]
            return input_images, azimuth_one_hot

    # def get_randomly_selected_models(self, batch_size):
    #     model_list = np.random.choice(self.model_list, batch_size)
    #     return model_list
    #
    # def get_batched_data_from_model_list(self, model_list):
    #     batch_size = len(model_list)
    #     input_random_azimuths = np.random.randint(self.n_elevation, size=batch_size)
    #     input_random_elevations = np.random.randint(self.n_azimuth, size=batch_size)
    #     images = np.zeros((batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
    #     for i in range(batch_size):
    #         images[i] = self.get_image_from_info(model_list[i], input_random_azimuths[i], input_random_elevations[i])
    #     return images, input_random_azimuths, input_random_elevations
    def get_one_hot_encoded(self, poseinfo):
        input_elevations_one_hot = np.eye(self.n_elevation, dtype=np.float32)[poseinfo[0]]
        input_azimuths_one_hot = np.eye(self.n_azimuth, dtype=np.float32)[poseinfo[1]]
        target_elevations_one_hot = np.eye(self.n_elevation, dtype=np.float32)[poseinfo[2]]
        target_azimuths_one_hot = np.eye(self.n_azimuth, dtype=np.float32)[poseinfo[3]]
        return (input_elevations_one_hot, input_azimuths_one_hot, target_elevations_one_hot, target_azimuths_one_hot)

    def get_batched_data_from_info(self, model_names, input_azimuths, target_azimuths):
        # load new model
        batch_size = len(model_names)
        input_random_elevations = np.random.randint(self.n_elevation, size=batch_size)
        input_random_azimuths = np.array(input_azimuths, dtype=np.int)
        target_random_elevations = np.random.randint(self.n_elevation, size=batch_size)
        target_random_azimuths = np.array(target_azimuths, dtype=np.int)
        input_images = np.zeros((batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
        target_images = np.zeros((batch_size, self.image_size, self.image_size, 3), dtype=np.float32)

        for i in range(batch_size):
            model_name = model_names[i]

            input_image = self.get_image_from_info(model_name, input_random_azimuths[i], input_random_elevations[i])
            target_image = self.get_image_from_info(model_name, target_random_azimuths[i],
                                                    target_random_elevations[i])

            input_images[i] = input_image
            target_images[i] = target_image

        return input_images, target_images, (input_random_elevations, input_random_azimuths, target_random_elevations, target_random_azimuths)


    def get_batched_data(self, batch_size=32, single_model=True, model_name=None, verbose=False):
        # load new model
        if single_model and model_name is None and (self.data_per_model_counter >= self.data_per_model or self.data_per_model == -1):
            self.current_model = random.choice(self.model_list)
            if self.load_all_first:
                self.load_all_images_of_single_model(self.current_model)
            elif self.use_cache:
                self.clear_image_cache()

            print(self.data_per_model_counter, " Counters", self.current_model, "loaded")
            self.data_per_model_counter = 0

        self.data_per_model_counter += batch_size

        input_random_elevations = np.random.randint(self.n_elevation, size=batch_size)
        input_random_azimuths = np.random.randint(self.n_azimuth, size=batch_size)
        target_random_elevations = np.random.randint(self.n_elevation, size=batch_size)
        target_random_azimuths = np.random.randint(self.n_azimuth, size=batch_size)
        model_names = []
        if self.load_all_first and single_model and model_name is None:
            input_images = self.all_images_for_single_model[input_random_elevations, input_random_azimuths]
            target_images = self.all_images_for_single_model[target_random_elevations, target_random_azimuths]
        else:
            input_images = np.zeros((batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
            target_images = np.zeros((batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
            if model_name is None:
                model_name = self.current_model

            for i in range(batch_size):
                if not single_model:
                    model_name = random.choice(self.model_list)
                    model_names.append(model_name)

                input_image = self.get_image_from_info(model_name, input_random_azimuths[i], input_random_elevations[i])
                target_image = self.get_image_from_info(model_name, target_random_azimuths[i], target_random_elevations[i])

                input_images[i] = input_image
                target_images[i] = target_image
        if verbose:
            print(model_names)
        if verbose:
            print(input_random_azimuths, target_random_azimuths)

        return input_images, target_images, (input_random_elevations, input_random_azimuths, target_random_elevations, target_random_azimuths)
        # else:
        #
        #     input_elevations_one_hot = np.eye(self.n_elevation, dtype=np.float32)[input_random_elevations]
        #     input_azimuths_one_hot = np.eye(self.n_azimuth, dtype=np.float32)[input_random_azimuths]
        #     target_elevations_one_hot = np.eye(self.n_elevation, dtype=np.float32)[target_random_elevations]
        #     target_azimuths_one_hot = np.eye(self.n_azimuth, dtype=np.float32)[target_random_azimuths]
        #
        #     return input_images, target_images, (input_elevations_one_hot, input_azimuths_one_hot, target_elevations_one_hot, target_azimuths_one_hot)

        #
        # if sin_cos:
        #     azimuth_theta = np.interp(input_random_elevations, xp=[0, self.n_azimuth], fp=[0, 2*np.pi])
        #     azimuth_cos = np.cos(azimuth_theta)
        #     azimuth_sin = np.cos(azimuth_theta)
        #
        #     if self.n_elevation < 1:
        #         elev_theta = np.interp(input_random_elevations, xp=[0, self.n_elevation - 1], fp=[-np.deg2rad(30), np.deg2rad(30)])
        #     else:
        #         elev_theta = np.zeros(input_random_elevations.shape)
        #     elev_cos = np.cos(elev_theta)
        #     elev_sin = np.sin(elev_theta)
        #
        #     return input_images, target_images,
        # else:
        #     input_poses = np.concatenate((input_elevations_one_hot, input_azimuths_one_hot, target_elevations_one_hot, target_azimuths_one_hot), axis=1)
        #
        # if not_concatenate:
        #     return input_images, input_poses, target_images, [input_elevations_one_hot, input_azimuths_one_hot,
        #                                        target_elevations_one_hot, target_azimuths_one_hot]
        #
        # return input_images, input_poses, target_images

    # def get_batched_data(self, batch_size=32, single_model=True, model_name=None):
    #     if self.load_all_first:
    #         return self.get_batched_data_all(batch_size, model_name)
    #     else:
    #         return self.get_batched_data_single(batch_size, single_model, model_name)

    # def get_batched_data_single(self, batch_size=32, single_model=True, model_name=None):
    #     pose_size = self.n_azimuth + self.n_elevation
    #
    #     input_images = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
    #     input_poses = np.zeros((batch_size, pose_size), dtype=np.float32)
    #     target_images = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
    #     target_poses = np.zeros((batch_size, pose_size), dtype=np.float32)
    #
    #     if self.data_per_model_counter >= self.data_per_model or self.data_per_model == -1:
    #         if self.use_cache:
    #             self.clear_image_cache()
    #         self.current_model = random.choice(self.model_list)
    #         self.data_per_model_counter = 0
    #         print(self.data_per_model_counter, " Counters", self.current_model, "loaded")
    #
    #     model_name = self.current_model
    #     for i in range(batch_size):
    #         if not single_model:
    #             model_name = random.choice(self.model_list)
    #         input_image, input_image_pose = self.get_image_pose_tuple(model_name)
    #         target_image, target_image_pose = self.get_image_pose_tuple(model_name)
    #
    #         input_images[i] = input_image #np.array(input_image)
    #         target_images[i] = target_image #np.array(target_image)
    #
    #         input_poses[i] = input_image_pose  # np.concatenate((input_image_pose, target_image_pose))
    #         target_poses[i] = target_image_pose  # np.concatenate((input_image_pose, target_image_pose))
    #
    #     input_images = input_images / 255.0
    #     target_images = target_images / 255.0
    #     input_poses = np.concatenate((input_poses, target_poses), axis=1)
    #     # print(input_poses.shape)
    #     return input_images, input_poses, target_images

    def get_image_pose_tuple(self, model_name=None):
        if model_name is None:
            model_name = random.choice(self.model_list)
        el = random.randint(0, 2)
        az = random.randint(0, 17)

        p_el = np.zeros(3)
        p_az = np.zeros(18)
        p_el[el] = 1
        p_az[az] = 1
        p = np.concatenate((p_el, p_az))

        file_name = "%s_%d_%d" % (model_name, az * 2, el * 10)
        image = self.data[file_name]['image']
        return image, p

    # def get_batched_data(self, batch_size=32, single_model=True, model_name=None):
    #     pose_size = self.n_azimuth + self.n_elevation
    #     pose_size *= 2
    #
    #     # input_images = np.zeros((batch_size, 256, 256, 3), dtype=np.uint8)
    #     # input_poses = np.zeros((batch_size, pose_size), dtype=np.uint8)
    #     # target_images = np.zeros((batch_size, 256, 256, 3), dtype=np.uint8)
    #
    #     if model_name is None:
    #         model_name = random.choice(self.model_list)
    #     # for i in range(batch_size):
    #     #     if not single_model:
    #     #         model_name = random.choice(self.model_list)
    #     #     input_image, input_image_pose = self.get_image_pose_tuple(model_name)
    #     #     target_image, target_image_pose = self.get_image_pose_tuple(model_name)
    #     #
    #     #     input_images[i] = np.array(input_image, dtype=np.uint8)
    #     #     target_images[i] = np.array(target_image, dtype=np.uint8)
    #     #
    #     #     input_poses[i] = np.concatenate((input_image_pose, target_image_pose))
    #     # import multiprocessing
    #     #
    #     # pool = multiprocessing.Pool()
    #     # input_images, input_poses = pool.map(self.get_image_pose_tuple, [model_name] * batch_size)
    #     # target_images, target_poses = pool.map(self.get_image_pose_tuple, [model_name] * batch_size)
    #
    #     input_images, input_poses = tf.numpy_function(
    #         self.get_image_pose_tuple, inp=[model_name] * batch_size,
    #         Tout=[np.float32, np.float32],
    #         name='func_hp'
    #     )
    #
    #     target_images, target_poses = tf.numpy_function(
    #         self.get_image_pose_tuple, inp=[model_name] * batch_size,
    #         Tout=[np.float32, np.float32],
    #         name='func_hp'
    #     )
    #     print(input_images.shape)
    #     print(type(input_images))
    #
    #     input_images.set_shape([batch_size, 256, 256, 3])
    #     target_images.set_shape([batch_size, 256, 256, 3])
    #     input_poses.set_shape([batch_size, pose_size])
    #     target_poses.set_shape([batch_size, pose_size])
    #
    #     input_poses = tf.concat([input_poses, target_poses], 1)
    #
    #     input_images = input_images / 255.0
    #     target_images = target_images / 255.0
    #
    #     print(input_images[0].dtype)
    #     #sess = tf.Session()
    #     # from keras.backend.tensorflow_backend import get_session
    #     #
    #     # sess = get_session()
    #     # with sess.as_default():
    #     #     input_images = input_images.eval()
    #     #     input_poses = input_poses.eval()
    #     #     target_images = target_images.eval()
    #     #
    #     return input_images, input_poses, target_images
    #     #return input_images.numpy(), input_poses.numpy(), target_images.numpy()
    #     #return K.eval(input_images), K.eval(input_poses), K.eval(target_images)
    #
    # def get_batched_data_old(self, batch_size=32, single_model=True, model_name=None):
    #     model_name = random.choice(self.model_list)
    #     input_ops = {}
    #     data_id = [model_name] * batch_size
    #
    #     # single operations
    #     with tf.device("/cpu:0"), tf.name_scope('inputs'):
    #         input_ops['id'] = tf.train.string_input_producer(
    #             tf.convert_to_tensor(data_id), capacity=128
    #         ).dequeue(name='input_ids_dequeue')
    #         m, p = self.get_image_pose_tuple(id)
    #
    #         def load_fn(id):
    #             # image [h, w, c*n]
    #             # pose [p, n]
    #             image, pose = self.get_image_pose_tuple(id)
    #             return (id, image.astype(np.float32), pose.astype(np.float32))
    #
    #         input_ops['id'], input_ops['image'], input_ops['camera_pose'] = tf.py_func(
    #             load_fn, inp=[input_ops['id']],
    #             Tout=[tf.string, tf.float32, tf.float32],
    #             name='func_hp'
    #         )
    #
    #         input_ops['id'].set_shape([])
    #         input_ops['image'].set_shape(list(m.shape))
    #         input_ops['camera_pose'].set_shape(list(p.shape))
    #
    #
    #     num_threads = 16
    #     # batchify
    #     capacity = 2 * batch_size * num_threads
    #     min_capacity = min(int(capacity * 0.75), 1024)
    #
    #     batch_ops = tf.train.batch(
    #         input_ops,
    #         batch_size=batch_size,
    #         num_threads=num_threads,
    #         capacity=capacity,
    #     )
    #
    #     input_images = batch_ops['image']
    #     input_poses = batch_ops['camera_pose']
    #
    #     target_images = batch_ops['image']
    #     input_poses = batch_ops['camera_pose']
    #
    #     return input_images, input_poses, target_images
    #
    #
    #
    #     # pose_size = self.n_azimuth + self.n_elevation
    #     # pose_size *= 2
    #     #
    #     # input_images = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
    #     # input_poses = np.zeros((batch_size, pose_size), dtype=np.float32)
    #     # target_images = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
    #     #
    #     # if model_name is None:
    #     #     model_name = random.choice(self.model_list)
    #     # for i in range(batch_size):
    #     #     if not single_model:
    #     #         model_name = random.choice(self.model_list)
    #     #     input_image, input_image_pose = self.get_image_pose_tuple(model_name)
    #     #     target_image, target_image_pose = self.get_image_pose_tuple(model_name)
    #     #
    #     #     input_images[i] = np.array(input_image)
    #     #     target_images[i] = np.array(target_image)
    #     #
    #     #     input_poses[i] = np.concatenate((input_image_pose, target_image_pose))
    #     #
    #     # # import multiprocessing
    #     # # pool = multiprocessing.Pool()
    #     # # results1 = pool.map(self.get_image_pose_tuple, [model_name] * batch_size)
    #     # # results2 = pool.map(self.get_image_pose_tuple, [model_name] * batch_size)
    #     # #
    #     # input_images = input_images / 255.0
    #     # target_images = target_images / 255.0
    #     #
    #     # print(input_images[0].dtype)
    #     #
    #     # return input_images, input_poses, target_images


from keras.utils import Sequence


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, model, data_container: DataContainer, need_mask = False, data_type = 'default'):
        self.data_container = data_container

        self.n_elevation = 3
        self.n_azimuth = 18

        'Initialization'
        self.batch_size = 32
        self.max_iter = 100000
        self.pixel_normalizer = model.pixel_normalizer
        self.pixel_normalizer_reverse = model.pixel_normalizer_reverse
        self.need_mask = need_mask
        self.single_model = True
        self.data_type = data_type

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.max_iter

    def __getitem__(self, index):
        from ops import get_background
        # Generate data
        if self.data_container.data is None:
            self.data_container.load_h5_data()
            # self.data = h5py.File(self.file_path, 'r', swmr=True)
            # self.data.visit()
            # print(len(self.data.iter))
            # a = self.data[0:10]

        if self.data_type == 'azimuth':
            x, y = self.data_container.get_batched_data_labeled(batch_size=self.batch_size)
            return x, y
        elif self.data_type == 'model_type':
            x, y = self.data_container.get_batched_data_labeled(batch_size=self.batch_size)



        input_images, input_poses, target_images = self.data_container.get_batched_data(batch_size=self.batch_size, single_model=self.single_model
                                                                                        )

        input_images = self.pixel_normalizer(input_images)
        # X = {'image_input': x, 'viewpoint_input': y},
        # Y = {'main_output': z, 'mask_output': get_background(z)}
        X = [input_images, input_poses]

        if self.need_mask:
            target_masks = get_background(target_images)
            target_images = self.pixel_normalizer(target_images)
            Y = [target_images, target_masks]
        else:
            target_images = self.pixel_normalizer(target_images)
            Y = [target_images]

        return X, Y

    def on_epoch_end(self):
        pass


class DataGeneratorForSequential(Sequence):
    'Generates data for Keras'

    def __init__(self, model, data_container: DataContainer, afn, need_mask=True):
        self.data_container = data_container

        self.n_elevation = 3
        self.n_azimuth = 18

        'Initialization'
        self.batch_size = 32
        self.max_iter = 100000
        self.pixel_normalizer = model.pixel_normalizer
        self.pixel_normalizer_reverse = model.pixel_normalizer_reverse
        self.need_mask = need_mask
        self.afn = afn

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.max_iter

    def __getitem__(self, index):
        from ops import get_background
        # Generate data
        if self.data_container.data is None:
            self.data_container.load_h5_data()
            # self.data = h5py.File(self.file_path, 'r', swmr=True)
            # self.data.visit()
            # print(len(self.data.iter))
            # a = self.data[0:10]

        input_images, input_poses, target_images = self.data_container.get_batched_data(batch_size=self.batch_size)
        input_images = self.pixel_normalizer(input_images)
        # X = {'image_input': x, 'viewpoint_input': y},
        # Y = {'main_output': z, 'mask_output': get_background(z)}
        X = [input_images, input_poses]

        if self.need_mask:
            target_masks = get_background(target_images)
            target_images = self.pixel_normalizer(target_images)
            Y = [target_images, target_masks]
        else:
            target_images = self.pixel_normalizer(target_images)
            Y = [target_images]

        return X, Y

    def on_epoch_end(self):
        pass

    # def get_batched_data(self, batch_size=32, single_model=True, model_name=None):
    #     pose_size = self.n_azimuth + self.n_elevation
    #     pose_size *= 2
    #
    #     input_images = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
    #     input_poses = np.zeros((batch_size, pose_size), dtype=np.float32)
    #     target_images = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
    #
    #     if model_name is None:
    #         model_name = random.choice(self.model_list)
    #     for i in range(batch_size):
    #         if not single_model:
    #             model_name = random.choice(self.model_list)
    #         input_image, input_image_pose = self.get_image_pose_tuple(model_name)
    #         target_image, target_image_pose = self.get_image_pose_tuple(model_name)
    #
    #         input_images[i] = np.array(input_image)
    #         target_images[i] = np.array(target_image)
    #
    #         input_poses[i] = np.concatenate((input_image_pose, target_image_pose))
    #
    #     # import multiprocessing
    #     # pool = multiprocessing.Pool()
    #     # results1 = pool.map(self.get_image_pose_tuple, [model_name] * batch_size)
    #     # results2 = pool.map(self.get_image_pose_tuple, [model_name] * batch_size)
    #     #
    #     input_images = input_images / 255.0
    #     target_images = target_images / 255.0
    #     return input_images, input_poses, target_images
    #
    # def find_all_models(self, data, model_number_limit):
    #     model_set = set()
    #     for k in data.keys():
    #         model_name = k.split("_")[0]
    #         model_set.add(model_name)
    #     self.model_list = list(model_set)
    #     if model_number_limit != - 1:
    #         self.model_list = self.model_list[0:min(model_number_limit, len(self.model_list))]
    #
    # def get_image_pose_tuple(self, model_name=None):
    #     if model_name is None:
    #         model_name = random.choice(self.model_list)
    #     el = random.randint(0, 2)
    #     az = random.randint(0, 17)
    #
    #     p_el = np.zeros(3)
    #     p_az = np.zeros(18)
    #     p_el[el] = 1
    #     p_az[az] = 1
    #     p = np.concatenate((p_el, p_az))
    #
    #     file_name = "%s_%d_%d" % (model_name, az * 2, el * 10)
    #     image = self.data[file_name]['image']
    #     return image, p
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from model.model_Tatarchenko15_attention import ModelTatarchenko15Attention
from model.model_Zhou16_attention import ModelZhou16Attention
from model.model_interface import ModelInterface
from data_container import *
import json
import multiprocessing
import os
import glob
import collections
import pandas as pd
from test_utils import *

dataset = None
current_test_input_images = None
current_test_target_images = None
current_test_poses = None


def initialize_tensorflow():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)


def build_model_from_dictionary(data: DataLoader, **kwargs):
    model_type = kwargs["model_type"]
    model_class = ModelInterface
    if model_type == 't':
        model_class = ModelTatarchenko15Attention
    elif model_type == 'z':
        model_class = ModelZhou16Attention

    attention_strategy = kwargs.get("attention_strategy", None)
    attention_strategy_details = kwargs.get("attention_strategy_details", None)
    random_seed_index = kwargs.get("random_seed_index", None)
    image_size = kwargs.get("image_size", 256)
    k = kwargs.get("k", 2)

    pose_input_size = None
    if data.name == 'kitti' or data.name == 'synthia':
        pose_input_size = data.pose_size

    model = model_class(
        image_size=image_size,
        attention_strategy=attention_strategy,
        attention_strategy_details=attention_strategy_details,
        additional_name=random_seed_index,
        pose_input_size=pose_input_size,
        k=k
    )

    return model


def find_load_model_in_folder(model, parent_folder, dataset_name):
    print(model.name)
    target_name = "%s/%s_%s*/*.h5" % (parent_folder, model.name, dataset_name)
    files = glob.glob(target_name)
    print(target_name)
    if len(files) > 1:
        min_file = None
        min_len = 100000
        for f in files:
            s = len(f.split("_"))
            if s < min_len:
                min_len = s
                min_file = f
        load_file = min_file
    else:
        load_file = files[0]
    return load_file


def load_dataset_from_config(**kwargs):
    dataset_name = kwargs["dataset"]
    dataset_format = kwargs["dataset_format"]
    image_size = kwargs.get("image_size", 256)
    train_or_test = kwargs.get("train_or_test", "train")
    if dataset_name == "kitti" or dataset_name == "synthia":
        if dataset_format == "npy":
            return SceneDataLoaderNumpy(dataset_name, image_size=image_size)
        else:
            return SceneDataLoader(dataset_name, image_size=image_size)
    elif dataset_name == "car" or dataset_name == "chair":
        return ObjectDataLoaderNumpy(dataset_name, image_size=image_size, train_or_test=train_or_test)


def train_single_model(x):
    i, gpu_id, config_file_name = x
    kwargs = json.load(open(config_file_name))
    ith_model_info = kwargs["model_list"][i]
    model = build_model_from_dictionary(dataset, **ith_model_info)
    print("model constructed!")

    additional_name = kwargs.get("additional_name", None)

    if additional_name is not None:
        random.seed(additional_name * 4219 + 123)
        np.random.seed(additional_name * 4219 + 123)
    else:
        random.seed(1000)
        np.random.seed(1000)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    initialize_tensorflow()

    model.train(dataset, **kwargs)


def train_all_using_multiprocessing(config_file_name):
    global dataset
    print("start to load dataset")
    config = json.load(open(config_file_name))
    model_counts = len(config["model_list"])
    dataset = load_dataset_from_config(**config)
    print("dataset loading finished")

    available_gpu_ids = config["available_gpu_ids"]
    gpu_ids = [available_gpu_ids[i % len(available_gpu_ids)] for i in range(model_counts)]
    train_infos = [(i, gpu_ids[i], config_file_name) for i in range(model_counts)]

    i = 0
    k = config.get("multiprocess_max", model_counts)

    print("start multiprocessing training")
    while i < model_counts:
        with multiprocessing.Pool(k) as p:
            p.map(train_single_model, train_infos[i:min(i + k, model_counts)], chunksize=1)
        i += k


def test_single_model(x):
    i, gpu_id, config_file_name = x
    kwargs = json.load(open(config_file_name))
    ith_model_info = kwargs["model_list"][i]
    model = build_model_from_dictionary(dataset, **ith_model_info)
    try:
        print("model constructed!")

        random.seed(883222)
        np.random.seed(883222)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        initialize_tensorflow()

        parent_folder = kwargs["parent_folder"]
        load_file = kwargs.get("load_file", find_load_model_in_folder(model, parent_folder, dataset.name))
        model.build_model()
        model.load_model(load_file)

        batch_size = kwargs.get("batch_size", 16)
        test_method = kwargs.get("test_method", "exhaustive")
        mae_all = None
        ssim_all = None
        if dataset.name == 'kitti' or dataset.name == 'synthia':
            if test_method == 'exhaustive':
                mae, ssim, mae_all, ssim_all = test_for_all_scenes(dataset, model, batch_size=batch_size)
            else:
                mae, ssim = test_for_random_scene(dataset, model, N=kwargs.get("max_iter", 20000), batch_size=batch_size)
        else:
            if test_method == 'exhaustive':
                mae, ssim, mae_all, ssim_all = test_for_all_models_thorough_per_model2(dataset, model,  batch_size=batch_size)
            else:
                mae, ssim = test_for_random_scene(dataset, model, N=kwargs.get("max_iter", 20000), batch_size=batch_size)

        return mae, ssim, mae_all, ssim_all, model.name
    except Exception as ex:
        print(ex)
        return 0, 0, model.name


def test_all_using_multiprocessing(config_file_name):
    global dataset
    config = json.load(open(config_file_name))
    model_counts = len(config["model_list"])
    config["train_or_test"] = "test"
    dataset = load_dataset_from_config(**config)
    print("dataset loading finished")

    available_gpu_ids = config["available_gpu_ids"]
    gpu_ids = [available_gpu_ids[i % len(available_gpu_ids)] for i in range(model_counts)]
    train_infos = [(i, gpu_ids[i], config_file_name) for i in range(model_counts)]

    k = config.get("multiprocess_max", model_counts)

    with multiprocessing.Pool(k) as p:
        results = p.map(test_single_model, train_infos, chunksize=1)

    maes, ssims, mae_alls, ssim_alls, names = zip(*results)

    raw_data = collections.OrderedDict()
    raw_data['name'] = names
    raw_data['mae'] = maes
    raw_data['ssim'] = ssims
    df = pd.DataFrame(raw_data)
    df = df.set_index("name")

    mae_alls = np.array(mae_alls)
    ssim_alls = np.array(ssim_alls)
    diff_N = mae_alls.shape[1]
    mae_all_df = pd.DataFrame(mae_alls, index=names, columns=[i - (diff_N // 2) for i in range(diff_N)])
    ssim_all_df = pd.DataFrame(ssim_alls, index=names, columns=[i - (diff_N // 2) for i in range(diff_N)])

    result_export_folder = config["result_export_folder"]
    if not os.path.exists(result_export_folder):
        os.makedirs(result_export_folder)
    started_time_date = time.strftime("%Y%m%d_%H%M%S")
    df.to_csv("%s/%s_%s.csv" % (result_export_folder, "total_result", started_time_date))
    mae_all_df.to_csv("%s/%s_%s.csv" % (result_export_folder, "total_result_mae", started_time_date))
    ssim_all_df.to_csv("%s/%s_%s.csv" % (result_export_folder, "total_result_ssim", started_time_date))


def test_and_export_picture_for_single_model(x):
    i, gpu_id, config_file_name = x
    kwargs = json.load(open(config_file_name))
    ith_model_info = kwargs["model_list"][i]
    model = build_model_from_dictionary(dataset, **ith_model_info)

    try:
        print("model constructed!")

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        initialize_tensorflow()

        parent_folder = kwargs["parent_folder"]
        load_file = kwargs.get("load_file", find_load_model_in_folder(model, parent_folder, dataset.name))
        model.build_model()
        model.load_model(load_file)

        poseinfo_processed = model.process_pose_info(dataset, current_test_poses)
        pred_images = model.get_predicted_image((current_test_input_images, poseinfo_processed))

        #pred_image_tensor = tf.convert_to_tensor(pred_images, dtype=tf.float32)
        #target_image_original_tensor = tf.convert_to_tensor(target_image_original, dtype=tf.float32)

        #ssim_values = K.eval(ssim_custom(pred_image_tensor, target_image_original_tensor))
        #mae_values = K.eval(mae_custom(pred_image_tensor, target_image_original_tensor))

        return pred_images, None, None, model.name

    except Exception as ex:
        print(ex)
        return None, None, None, model.name


def test_and_export_picture_for_models_using_multiprocessing(config_file_name):
    global dataset, current_test_input_images, current_test_target_images, current_test_poses
    config = json.load(open(config_file_name))
    model_counts = len(config["model_list"])
    config["train_or_test"] = "test"
    dataset = load_dataset_from_config(**config)
    print("dataset loading finished")

    available_gpu_ids = config["available_gpu_ids"]
    gpu_ids = [available_gpu_ids[i % len(available_gpu_ids)] for i in range(model_counts)]
    train_infos = [(i, gpu_ids[i], config_file_name) for i in range(model_counts)]

    k = config.get("multiprocess_max", model_counts)

    target_scene_infos = config.get("target_scene_infos", None)
    target_scene_n = config.get("target_scene_n", 5)
    result_export_folder = config.get("result_export_folder", None)
    index_info = None

    if target_scene_infos is None:
        test_data, index_info = dataset.get_batched_data(target_scene_n, single_model=False, return_info=True, is_train=False)
        print(index_info)
    else:
        test_data = dataset.get_specific_data(target_scene_infos)
    current_test_input_images, current_test_target_images, current_test_poses = test_data

    with multiprocessing.Pool(k) as p:
        results = p.map(test_and_export_picture_for_single_model, train_infos, chunksize=1)

    images, maes, ssims, names = zip(*results)

    # 1. export images
    xs = []
    xs.append(np.concatenate(current_test_input_images, axis=0))
    xs.append(np.concatenate(current_test_target_images, axis=0))
    pred_image_temp = None
    for pred_image in images:
        if pred_image is not None:
            xs.append(np.concatenate(pred_image, axis=0))
        elif pred_image_temp is not None:
            xs.append(np.concatenate(np.zeros_like(pred_image_temp), axis=0))
        pred_image_temp = pred_image

    total_image = np.concatenate(tuple(xs), axis=1)

    if not os.path.exists(result_export_folder):
        os.makedirs(result_export_folder)
    started_time_date = time.strftime("%Y%m%d_%H%M%S")
    save_pred_images(total_image, "%s/%s_%s" % (result_export_folder, "total_images", started_time_date))

    # export model names
    raw_data = collections.OrderedDict()
    raw_data['name'] = names
    df = pd.DataFrame(raw_data)
    df = df.set_index("name")
    df.to_csv("%s/%s_%s.csv" % (result_export_folder, "total_images_models", started_time_date))

    if index_info is not None:
        if dataset.name == 'kitti' or dataset.name == 'synthia':
            scene_ids, input_ids, target_ids = zip(*index_info)
            raw_data = collections.OrderedDict()
            raw_data['scene_id'] = scene_ids
            raw_data['input_id'] = input_ids
            raw_data['target_id'] = target_ids
            df = pd.DataFrame(raw_data)
            df.to_csv("%s/%s_%s.csv" % (result_export_folder, "tested_samples_index_info", started_time_date), index=False)


def test_and_export_feature_map_for_single_model(x):
    i, gpu_id, config_file_name = x
    kwargs = json.load(open(config_file_name))
    ith_model_info = kwargs["model_list"][i]
    model = build_model_from_dictionary(dataset, **ith_model_info)

    #try:
    print("model constructed!")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    initialize_tensorflow()

    parent_folder = ith_model_info["parent_folder"]
    result_export_folder = kwargs["result_export_folder"]
    load_file = kwargs.get("load_file", find_load_model_in_folder(model, parent_folder, dataset.name))
    model.build_model()
    model.load_model(load_file)

    poseinfo_processed = model.process_pose_info(dataset, current_test_poses)
    current_test_data = (current_test_input_images, current_test_target_images, poseinfo_processed)
    feature_map = calculate_encoder_decoder_similarity(current_test_data, model)
    started_time_date = time.strftime("%Y%m%d_%H%M%S")
    print(feature_map.shape)
    save_pred_images(feature_map, "%s/%s_%s" % (result_export_folder, model.name, started_time_date))
    # except Exception as ex:
    #     print(ex)
    #     return


def test_and_export_feature_map_for_models_using_multiprocessing(config_file_name):
    global dataset, current_test_input_images, current_test_target_images, current_test_poses

    config = json.load(open(config_file_name))
    model_counts = len(config["model_list"])
    dataset = load_dataset_from_config(**config)
    print("dataset loading finished")

    available_gpu_ids = config["available_gpu_ids"]
    gpu_ids = [available_gpu_ids[i % len(available_gpu_ids)] for i in range(model_counts)]
    train_infos = [(i, gpu_ids[i], config_file_name) for i in range(model_counts)]

    k = config.get("multiprocess_max", model_counts)

    target_scene_infos = config.get("target_scene_infos", None)
    index_info = None

    if target_scene_infos is None:
        test_data, index_info = dataset.get_batched_data(1, single_model=False, return_info=True)
        print(index_info)
    else:
        test_data = dataset.get_specific_data(target_scene_infos)
    current_test_input_images, current_test_target_images, current_test_poses = test_data

    with multiprocessing.Pool(k) as p:
        p.map(test_and_export_feature_map_for_single_model, train_infos, chunksize=1)
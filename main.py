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


dataset = None


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

    pose_input_size = None
    if data.name == 'kitti' or data.name == 'synthia':
        pose_input_size = data.pose_size

    model = model_class(
        image_size=image_size,
        attention_strategy=attention_strategy,
        attention_strategy_details=attention_strategy_details,
        additional_name=random_seed_index,
        pose_input_size=pose_input_size
    )

    return model


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


def find_load_model_in_folder(model, parent_folder, dataset_name):
    print(model.name)
    files = glob.glob("%s/%s_%s*/*.h5" % (parent_folder, model.name, dataset_name))
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

from test_utils import test_for_random_scene


def test_single_model(x):
    i, gpu_id, config_file_name = x
    kwargs = json.load(open(config_file_name))
    ith_model_info = kwargs["model_list"][i]
    model = build_model_from_dictionary(dataset, **ith_model_info)
    print("model constructed!")

    random.seed(883222)
    np.random.seed(883222)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    initialize_tensorflow()

    parent_folder = kwargs["parent_folder"]
    load_file = kwargs.get("load_file", find_load_model_in_folder(model, parent_folder, dataset.name))
    model.build_model()
    model.load_model(load_file)

    mae, ssim = test_for_random_scene(dataset, model, N=kwargs.get("max_iter", 20000))

    return mae, ssim, model.name



def load_dataset_from_config(**kwargs):
    dataset_name = kwargs["dataset"]
    dataset_format = kwargs["dataset_format"]
    image_size = kwargs.get("image_size", 256)

    if dataset_name == "kitti" or dataset_name == "synthia":
        if dataset_format == "npy":
            return SceneDataLoaderNumpy(dataset_name, image_size=image_size)
        else:
            return SceneDataLoader(dataset_name, image_size=image_size)


def train_all_using_multiprocessing(config_file_name):
    global dataset
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


def test_all_using_multiprocessing(config_file_name):
    global dataset
    config = json.load(open(config_file_name))
    model_counts = len(config["model_list"])
    dataset = load_dataset_from_config(**config)
    print("dataset loading finished")

    available_gpu_ids = config["available_gpu_ids"]
    gpu_ids = [available_gpu_ids[i % len(available_gpu_ids)] for i in range(model_counts)]
    train_infos = [(i, gpu_ids[i], config_file_name) for i in range(model_counts)]

    k = config.get("multiprocess_max", model_counts)

    with multiprocessing.Pool(k) as p:
        results = p.map(test_single_model, train_infos, chunksize=1)

    maes, ssims, names = zip(*results)

    raw_data = collections.OrderedDict()
    raw_data['name'] = names
    raw_data['mae'] = maes
    raw_data['ssim'] = ssims

    df = pd.DataFrame(raw_data)
    df = df.set_index("name")
    model_export_folder = config["result_export_folder"]
    if not os.path.exists(model_export_folder):
        os.makedirs(model_export_folder)
    started_time_date = time.strftime("%Y%m%d_%H%M%S")
    df.to_csv("%s/%s_%s.csv" % (model_export_folder, "total_result", started_time_date))

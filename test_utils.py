import numpy as np
from data_container import DataLoader, SceneDataLoader, ObjectDataLoaderNumpy
from PIL import Image
import keras.backend as K
import tensorflow as tf


def align_input_output_image(inputs, target, pred):
    x1 = np.concatenate(inputs, axis=1)
    x2 = np.concatenate(target, axis=1)
    x3 = np.concatenate(pred, axis=1)

    xs = np.concatenate((x1, x2, x3), axis=0)
    return xs


def save_pred_images(images, file_path):
    x = images
    x *= 255
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    new_im = Image.fromarray(x)
    new_im.save("%s.png" % (file_path))


def test_few_models_and_export_image(model, data: DataLoader, file_name, folder_name, test_n=5,
                                     single_model=False):
    input_image_original, target_image_original, poseinfo = data.get_batched_data(test_n, single_model=single_model)
    poseinfo_processed = model.process_pose_info(data, poseinfo)
    pred_images = model.get_predicted_image((input_image_original, poseinfo_processed))
    images = align_input_output_image(input_image_original, target_image_original, pred_images)
    save_pred_images(images, "%s/%s" % (folder_name, file_name))

    return images


def ssim_custom(y_true, y_pred):
    return tf.image.ssim(y_pred, y_true, max_val=1.0, filter_sigma=0.5)


def mae_custom(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))


def test_for_random_scene(data: SceneDataLoader, model, N=20000, batch_size=32):
    mae = 0
    ssim = 0
    count = 0
    while count < N:
        input_image_original, target_image_original, pose_info = data.get_batched_data(batch_size=batch_size, is_train=False)
        pose_info_per_model = model.process_pose_info(data, pose_info)
        metrics = model.evaluate(input_image_original, target_image_original, pose_info_per_model)
        mae += metrics[1] * batch_size
        ssim += metrics[2] * batch_size
        count += batch_size

    mae /= count
    ssim /= count
    return mae, ssim


def test_for_all_scenes(data: SceneDataLoader, model, batch_size=16):
    scene_N = len(data.scene_list)
    difference_N = 2 * data.max_frame_difference + 1
    absolute_errors = np.zeros((difference_N, ), dtype=np.float32)
    ssim_errors = np.zeros((difference_N, ), dtype=np.float32)

    for difference in range(difference_N):
        for i in range(len(data.scene_list)):
            scene_id = data.scene_list[i]
            index = 0
            N = len(data.test_ids[scene_id])
            while index < N:
                M = min(index + batch_size, N)
                input_image_original, target_image_original, pose_info = data.get_batched_data_i_j(
                    scene_id, difference - data.max_frame_difference, index, M)
                pose_info_per_model = model.process_pose_info(data, pose_info)
                metrics = model.evaluate(input_image_original, target_image_original, pose_info_per_model)
                absolute_errors[difference] += metrics[1] * (M - index)
                ssim_errors[difference] += metrics[2] * (M - index)
                index += batch_size

    total_N = 0
    for scene_id in data.scene_list:
        total_N += len(data.test_ids[scene_id])

    absolute_errors /= total_N
    ssim_errors /= total_N

    absolute_errors_avg = np.mean(absolute_errors)
    ssim_errors_avg = np.mean(ssim_errors)

    return absolute_errors_avg, ssim_errors_avg, absolute_errors, ssim_errors


def test_for_all_objects(data: ObjectDataLoaderNumpy, model, batch_size=50):
    absolute_errors = np.zeros((18, 18))
    ssim_errors = np.zeros((18, 18))

    N = data.n_models
    for i in range(18):
        for j in range(18):
            print(i, j)
            index = 0
            while index < N:
                M = min(index + batch_size, N)
                input_image_original, target_image_original, pose_info = data.get_batched_data_i_j(i, j, index, M)
                pose_info_per_model = model.process_pose_info(data, pose_info)
                metrics = model.evaluate(input_image_original, target_image_original, pose_info_per_model)
                absolute_errors[i][j] += metrics[1] * (M - index)
                ssim_errors[i][j] += metrics[2] * (M - index)
                index += batch_size

    absolute_errors /= N
    ssim_errors /= N

    absolute_errors2 = np.zeros((18, ), dtype=np.float32)
    ssim_errors2 = np.zeros((18, ), dtype=np.float32)

    for i in range(18):
        for j in range(18):
            index = (18 + j - i) % 18
            absolute_errors2[index] += absolute_errors[i, j]
            ssim_errors2[index] += ssim_errors[i, j]

    absolute_errors2 = absolute_errors2 / 18
    ssim_errors2 = ssim_errors2 / 18

    mae = np.mean(absolute_errors2)
    ssim = np.mean(ssim_errors2)
    return mae, ssim, absolute_errors2, ssim_errors2

from matplotlib import pyplot as plt
from skimage.transform import resize


def show_feature_map(data: DataLoader, model, test_n=5):
    input_image_original, target_image_original, poseinfo = data.get_batched_data(1, single_model=False)
    s = K.get_session()
    ks = [16, 32, 64, 128]
    w = 4
    h = 4

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(input_image_original[0])
    plt.subplot(1, 3, 2)
    plt.imshow(target_image_original[0])
    plt.subplot(1, 3, 3)
    per_model_poseinfo = model.process_pose_info(data, poseinfo)

    pred_images = model.get_predicted_image((input_image_original, per_model_poseinfo))
    plt.imshow(pred_images[0])

    per_model_poseinfo = model.process_pose_info(data, poseinfo)
    flow = s.run(model.pred_flow,
                 feed_dict={
                     model.model.input[0]: input_image_original,
                     model.model.input[1]: per_model_poseinfo
                 })
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(flow[0,:,:,0])
    plt.subplot(1, 2, 2)
    plt.imshow(flow[0,:,:,1])

    plt.rcParams["figure.figsize"] = (8, 8)
    def f(type = 1):
        for k in reversed(ks):
            plt.figure()
            if type == 1:
                target = model.encoder_original_features[k]
            elif type == 2:
                target = model.decoder_original_features[k]
            else:
                target = model.decoder_rearranged_features[k]

            feature_maps = s.run(target,
                                 feed_dict={
                                     model.model.input[0]: input_image_original,
                                     model.model.input[1]: per_model_poseinfo
                                 })

            # f, axarr = plt.subplots(8, 8, figsize=(24, 24))
            ix = 1
            for i in range(h):
                for j in range(w):
                    ax = plt.subplot(w, h, ix)
                    img = feature_maps[0, :, :, ix - 1]
                    plt.imshow(img)
                    ix += 1

    f(1)
    f(2)
    f(3)


def put_image_in(large_image, image, x, y, size):
    large_image[x*size:x*size+size, y*size:y*size+size, :] = image


def normalize_image(x):
    return (x - x.min()) / (x.max() - x.min())


def show_feature_map(test_data, model):
    input_image_original, target_image_original, processed_pose_info = test_data
    s = K.get_session()
    ks = [16, 32, 64, 128]
    N = 128
    output_images = np.zeros((N * len(ks), N * 3, 1), dtype=np.float32)

    for k_i, k in enumerate(reversed(ks)):
        x_d = model.decoder_original_features[k]
        x_e = model.decoder_rearranged_features[k]
        x_e_0 = model.encoder_original_features[k]
        decoder_features, rearranged_encoder_features, encoder_features = s.run([x_d, x_e, x_e_0],
                             feed_dict={
                                 model.model.input[0]: input_image_original,
                                 model.model.input[1]: processed_pose_info
                             })
        decoder_features = decoder_features[0]
        rearranged_encoder_features = rearranged_encoder_features[0]
        encoder_features = encoder_features[0]
        do_abs = True
        if do_abs:
            decoder_features = np.abs(decoder_features)
            rearranged_encoder_features = np.abs(rearranged_encoder_features)
            encoder_features = np.abs(encoder_features)

        do_mean = True
        if do_mean:
            mean_decoder_features = np.mean(decoder_features, axis=2, keepdims=True)
            mean_rearranged_encoder_features = np.mean(rearranged_encoder_features, axis=2, keepdims=True)
            encoder_features = np.mean(encoder_features, axis=2, keepdims=True)
        else:
            mean_decoder_features = decoder_features[:,:,0:1]
            mean_rearranged_encoder_features = rearranged_encoder_features[:,:,0:1]
            encoder_features = encoder_features[:,:,0:1]


        do_resize = True
        if do_resize:
            t = 128
            mean_decoder_features = resize(mean_decoder_features, (t, t)).astype(np.float32)
            mean_rearranged_encoder_features = resize(mean_rearranged_encoder_features, (t, t)).astype(np.float32)
            encoder_features = resize(encoder_features, (t, t)).astype(np.float32)
        do_normalize = True
        if do_normalize:
            mean_decoder_features = normalize_image(mean_decoder_features)
            encoder_features = normalize_image(encoder_features)
            mean_rearranged_encoder_features = normalize_image(mean_rearranged_encoder_features)

        put_image_in(output_images, encoder_features, k_i, 0, N)
        put_image_in(output_images, mean_rearranged_encoder_features, k_i, 1, N)
        put_image_in(output_images, mean_decoder_features, k_i, 2, N)

    output_images = output_images[:,:,0]
    plt.figure()
    plt.imshow(output_images)
    plt.show()
    return output_images
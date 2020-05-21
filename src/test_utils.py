import numpy as np
from data_container import DataLoader
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
    # x = np.concatenate(images, axis=1)
    x = images
    x *= 255
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    new_im = Image.fromarray(x)
    new_im.save("%s.png" % (file_path))


def test_few_models_and_export_image(model, data: DataLoader, iteration, folder_name, test_n=5,
                                     single_model=False):
    input_image_original, target_image_original, poseinfo = data.get_batched_data(test_n, single_model=single_model)
    poseinfo_processed = model.process_pose_info(data, poseinfo)
    pred_images = model.get_predicted_image((input_image_original, poseinfo_processed))
    images = align_input_output_image(input_image_original, target_image_original, pred_images)
    save_pred_images(images, "%s/%d" % (folder_name, iteration))


def ssim_custom(y_true, y_pred):
    return tf.image.ssim(y_pred, y_true, max_val=1.0, filter_sigma=0.5)
def mae_custom(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))

# def ssim_custom(y_true, y_pred):
#     result = tf.image.ssim(y_pred, y_true, max_val=1.0, filter_sigma=0.5)
#     result = K.print_tensor(result, message='losses_ssim')
#     return result
#
#
# def mae_custom(y_true, y_pred):
#     result = mean_absolute_error(y_true, y_pred)
#     result = K.print_tensor(result, message='losses_l1')
#     return result
#
# def test_for_all_models_thorough_per_model2(data: DataContainer, model):
#     #absolute_errors = np.zeros((len(data.model_list), 18, 18))
#     #ssim_errors = np.zeros((len(data.model_list), 18, 18))
#
#     absolute_errors = np.zeros((18, 18))
#     ssim_errors = np.zeros((18, 18))
#
#     N = len(data.model_list)
#
#     for i in range(18):
#         for j in range(18):
#             index = 0
#             batch_size = 32
#             while index < N:
#                 M = min(index + batch_size)
#                 input_image_original, target_image_original, pose_info = data.get_batched_data_i_j(i, j, index, M)
#                 pose_info_per_model = model.process_pose_info(data, pose_info)
#                 metrics = model.evaluate([input_image_original, pose_info], target_image_original, verbose=False)
#                 absolute_errors[i][j] += metrics[1] * (M - index)
#                 ssim_errors[i][j] += metrics[2] * (M - index)
#
#     absolute_errors /= N
#     ssim_errors /= N
#
#     #
#     # for k, model_name in enumerate(data.model_list):
#     #     print(model_name)
#     #     for m in range(18):
#     #         input_image_original, target_image_original, pose_info = data.get_batched_data_single(start_angle=m, model_name=model_name)
#     #
#     #         # metrics = ms[i].evaluate([input_image_original, pose_info], target_image_original, verbose=False)
#     #         pose_info_per_model = model.process_pose_info(data, pose_info)
#     #
#     #         for j in range(18):
#     #             metrics = model.evaluate(input_image_original[j:j+1], target_image_original[j:j+1], pose_info_per_model[j:j+1])
#     #
#     #             absolute_errors[k][m][j] = metrics[1]
#     #             ssim_errors[k][m][j] = metrics[2]
#
#     mae = np.mean(absolute_errors)
#     ssim = np.mean(ssim_errors)
#
#     return mae, ssim


def test_for_all_models_thorough_per_model2(data: DataLoader, model, batch_size=50):
    #absolute_errors = np.zeros((len(data.model_list), 18, 18))
    #ssim_errors = np.zeros((len(data.model_list), 18, 18))

    absolute_errors = np.zeros((18, 18))
    ssim_errors = np.zeros((18, 18))

    N = len(data.model_list)
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
    mae = np.mean(absolute_errors)
    ssim = np.mean(ssim_errors)
    return mae, ssim, absolute_errors, ssim_errors


def test_for_all_models_thorough_per_model(data: DataLoader, model):
    absolute_errors = np.zeros((len(data.model_list), 18))
    ssim_errors = np.zeros((len(data.model_list), 18))

    for k, model_name in enumerate(data.model_list):
        print(model_name)
        for m in range(18):
            input_image_original, target_image_original, pose_info = data.get_batched_data_single(start_angle=m, model_name=model_name)
            pose_info_per_model = model.process_pose_info(data, pose_info)
            metrics = model.evaluate(input_image_original, target_image_original, pose_info_per_model)

            absolute_errors[k][m] = metrics[1]
            ssim_errors[k][m] = metrics[2]
            print(metrics[1].shape)

    mae = np.mean(absolute_errors)
    ssim = np.mean(ssim_errors)

    return mae, ssim

from matplotlib import pyplot as plt
from skimage.transform import resize


# def show_feature_map(data: DataContainer, model, test_n=5):
#     input_image_original, target_image_original, poseinfo = data.get_batched_data(1, single_model=False)
#     pred_images_list = []
#     errors = []
#     from keras.losses import mean_squared_error
#     y_true = K.variable(target_image_original)
#     attention_maps = {}
#     feature_maps = {}
#
#     s = K.get_session()
#     #tf.enable_eager_execution()
#     ks = [16, 32, 64, 128]
#     w = 4
#     h = 4
#     plt.figure()
#     plt.subplot(1, 3, 1)
#     plt.imshow(input_image_original[0])
#     plt.subplot(1, 3, 2)
#     plt.imshow(target_image_original[0])
#     plt.subplot(1, 3, 3)
#     per_model_poseinfo = model.process_pose_info(data, poseinfo)
#     pred_images = model.get_predicted_image((input_image_original, per_model_poseinfo))
#     plt.imshow(pred_images[0])
#
#     per_model_poseinfo = model.process_pose_info(data, poseinfo)
#     flow = s.run(model.pred_flow,
#                  feed_dict={
#                      model.model.input[0]: input_image_original,
#                      model.model.input[1]: per_model_poseinfo
#                  })
#     plt.figure()
#     plt.subplot(1, 2, 1)
#     plt.imshow(flow[0,:,:,0])
#     plt.subplot(1, 2, 2)
#     plt.imshow(flow[0,:,:,1])
#
#     image_flow_tensors = {}
#     for k in ks:
#         a = resize(input_image_original[0], (k, k)).astype(np.float32)
#         b = resize(target_image_original[0], (k, k)).astype(np.float32)
#         c = resize(flow[0], (k, k)).astype(np.float32)
#         a = np.expand_dims(color.rgb2gray(a), 2).astype(np.float32)
#         b = np.expand_dims(color.rgb2gray(b), 2).astype(np.float32)
#         print(a.shape)
#         image_flow_tensors[k] = tf.convert_to_tensor(np.concatenate((a, b, c), axis=2))
#
#     N = 4
#
#     plt.rcParams["figure.figsize"] = (8, 8)
#     def f(do_input=True):
#         for k in reversed(ks):
#             plt.figure()
#             target = model.input_hidden_layers[k] if do_input else model.output_hidden_layers[k]
#             feature_maps = s.run(target,
#                                  feed_dict={
#                                      model.model.input[0]: input_image_original,
#                                      model.model.input[1]: per_model_poseinfo
#                                  })
#
#             # f, axarr = plt.subplots(8, 8, figsize=(24, 24))
#             ix = 1
#             ssim_values = np.zeros((N, w * h), dtype=np.float32)
#             for i in range(h):
#                 for j in range(w):
#                     ax = plt.subplot(w, h, ix)
#                     img = feature_maps[0, :, :, ix - 1]
#                     plt.imshow(img)
#
#                     for c in range(N):
#                         ssim_value = K.eval(
#                             ssim_custom(image_flow_tensors[k][:, :, c:c + 1],
#                                         tf.expand_dims(tf.convert_to_tensor(img), axis=2)))
#                         ssim_values[c][ix - 1] = ssim_value
#
#                     ix += 1
#             plt.show(block=False)
#             #print(ssim_values)
#             print(np.mean(ssim_values, axis=1))
#
#     f(True)
#     f(False)
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

def calculate_encoder_decoder_similarity(data: DataLoader, model, model_info=None):
    if model_info is not None:
        lst1, lst2, lst3 = zip(*model_info)
        test_n = len(lst1)
        input_image_original, target_image_original, poseinfo = data.get_batched_data_from_info(lst1,lst2,lst3)
    else:
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

    N = 128
    output_images = np.zeros((N * len(ks), N * 3, 1), dtype=np.float32)

    if 'zhou' in model.name:
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
    for k_i, k in enumerate(reversed(ks)):
        x_d = model.decoder_original_features[k]
        x_e = model.decoder_rearranged_features[k]
        x_e_0 = model.encoder_original_features[k]
        decoder_features, rearranged_encoder_features, encoder_features = s.run([x_d, x_e, x_e_0],
                             feed_dict={
                                 model.model.input[0]: input_image_original,
                                 model.model.input[1]: per_model_poseinfo
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

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(mean_decoder_features[:,:,0])
        plt.subplot(1, 3, 2)
        plt.imshow(mean_rearranged_encoder_features[:,:,0])
        plt.subplot(1, 3, 3)
        plt.imshow(encoder_features[:,:,0])


        A = tf.convert_to_tensor(mean_decoder_features)
        B = tf.convert_to_tensor(mean_rearranged_encoder_features)
        ssim_value = K.eval(tf.image.ssim(A, B, max_val=1))
        print(k, ssim_value)

        l1 = K.eval(K.mean(K.abs(A-B)))
        print(k, l1)
    return output_images

def calculate_encoder_decoder_similarity2(data: DataLoader, model, test_n=10):
    s = K.get_session()
    ks = [16, 32, 64, 128]
    ssims = {}
    l1s = {}
    for k in ks:
        ssims[k] = []
        l1s[k] = []

    for n in range(test_n):
        print(n)
        w = 4
        h = 4
        input_image_original, target_image_original, poseinfo = data.get_batched_data(1, single_model=False)
        per_model_poseinfo = model.process_pose_info(data, poseinfo)

        N = 128

        for k_i, k in enumerate(reversed(ks)):
            x_d = model.decoder_original_features[k]
            x_e = model.decoder_rearranged_features[k]
            x_e_0 = model.encoder_original_features[k]
            decoder_features, rearranged_encoder_features, encoder_features = s.run([x_d, x_e, x_e_0],
                                 feed_dict={
                                     model.model.input[0]: input_image_original,
                                     model.model.input[1]: per_model_poseinfo
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

            A = tf.convert_to_tensor(mean_decoder_features)
            B = tf.convert_to_tensor(mean_rearranged_encoder_features)
            ssim_value = K.eval(tf.image.ssim(A, B, max_val=1))
            l1 = K.eval(K.mean(K.abs(A-B)))
            ssims[k].append(ssim_value)
            l1s[k].append(l1)

    for k in ks:
        ssims[k] = np.array(ssims[k]).mean()
        l1s[k] = np.array(l1s[k]).mean()
    print(ssims, l1s)


import imageio
import time


def images_to_gif(data: DataLoader, model_list, n=5, video_name=None):
    started_time_date = time.strftime("%Y%m%d_%H%M%S")

    videos = [None] * len(model_list)
    video_0 = None
    for i in range(n):
        input_image_original, target_image_original, poseinfo = data.get_batched_data_single(start_angle=5)

        #print(input_image_original.shape)
        #print(input_image_original.shape)

        # pred_images = np.hstack((input_image_original, pred_images))
        video = np.copy(input_image_original)
        video = np.concatenate((video, target_image_original), axis=1)

        for j, model in enumerate(model_list):
            # input_image = model.pixel_normalizer(input_image_original)
            poseinfo_processed = model.process_pose_info(data, poseinfo)
            pred_images = model.get_predicted_image((input_image_original, poseinfo_processed))
            # video = videos[j]

            if video is None:
                video = pred_images
            else:
                video = np.concatenate((video, pred_images), axis=1)

        video *= 255
        video = np.clip(video, 0, 255)
        video = video.astype('uint8')

        if video_0 is None:
            video_0 = video
        else:
            video_0 = np.concatenate((video_0, video), axis=2)

    if video_name is None:
        video_name = "video"
    imageio.mimwrite('%s_%s_%s.mp4' % (data.name, video_name, started_time_date), video_0, fps=2)
from data_container import *
from model.model_appearence_flow import *
from model.model_sequential import ModelSequential
from model.model_interface import ModelInterface
from model.model_discriminator import Discriminator
from model.model_identity_discriminator import IdentityDiscriminator
from model.model_vgg_feature import VGG16Model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import CSVLogger
import time
from matplotlib import pyplot as plt
import keras.backend as K
import keras.losses

import keras
import pydot as pyd
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import os
from pandas import DataFrame

keras.utils.vis_utils.pydot = pyd


def visualize_model(model):
    return SVG(model_to_dot(model).create(prog='dot', format='svg'))


def initialize_tensorflow():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras


def model_with_discriminator(model: ModelInterface):
    discriminator = Discriminator()
    optimizer_d = Adam(lr=0.01, beta_1=0.9)

    discriminator.get_model().compile(optimizer=optimizer_d, loss='binary_crossentropy')
    discriminator.get_model().trainable = False

    model.get_model().name = "generator"
    discriminator.get_model().name = "gan_output"
    gan_output = discriminator.get_model()(model.get_model().output)
    gan = Model(inputs=model.get_model().input, outputs=[gan_output, model.get_model().output])
    gan.summary()
    return gan


def save_pred_images(images, file_path):
    # x = np.concatenate(images, axis=1)
    x = images
    x *= 255
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    new_im = Image.fromarray(x)
    new_im.save("%s.png" % (file_path))


def align_input_output_image(inputs, target, pred):
    x1 = np.concatenate(inputs, axis=1)
    x2 = np.concatenate(target, axis=1)
    x3 = np.concatenate(pred, axis=1)

    xs = np.concatenate((x1, x2, x3), axis=0)
    return xs

def test_few_models_and_export_image(model: ModelInterface, data: DataLoader, iteration, folder_name, test_n=5,
                                     single_model=False):
    input_image_original, target_image_original, azelinfo = data.get_batched_data(test_n, single_model=single_model, one_hot=True)
    pred_images = model.get_predicted_image((input_image_original, azelinfo))
    images = align_input_output_image(input_image_original, target_image_original, pred_images)
    save_pred_images(images, "%s/%d" % (folder_name, iteration))


def train_with_discriminator(model: ModelInterface,
                             data: DataLoader,
                             max_iter=100000,
                             save_model=True,
                             write_log=True,
                             gan_weight=1,
                             export_image_per=1000,
                             single_model=False,
                             batch_size=32):
    from keras.optimizers import Adam
    discriminator_optimizer = Adam(lr=0.0001)

    discriminator = Discriminator(model.image_size)

    discriminator.build_model()
    discriminator.get_model().compile(optimizer=discriminator_optimizer, loss={'gan_loss': 'binary_crossentropy'})
    discriminator.get_model().trainable = False

    model.get_model().name = "generator"
    image_input = keras.Input(shape=(model.image_size, model.image_size, 3), name='gan_image_input')
    viewpoint_input = keras.Input(shape=(42,), name='gan_viewpoint_input')

    generated_image = model.get_model()([image_input, viewpoint_input])

    gan_output = discriminator.get_model()(generated_image)

    vgg_model = VGG16Model(model.image_size)
    vgg_model.build_model()
    vgg_feature_output = vgg_model.get_model()(generated_image)

    gan = Model(inputs=[image_input, viewpoint_input], outputs=[generated_image] + gan_output + vgg_feature_output)
    # vgg_model.get_model().get_layer('block1_pool').output,
    # vgg_model.get_model().get_layer('block2_pool').output,
    # vgg_model.get_model().get_layer('block3_pool').output,
    # discriminator.get_model().get_layer('d_en_conv_0').output,
    # discriminator.get_model().get_layer('d_en_conv_1').output,
    # discriminator.get_model().get_layer('gan_loss').output
    # ])
    gan.summary()

    # visualize_model(gan)
    started_time_date = time.strftime("%Y%m%d_%H%M%S")

    gan_optimizer = Adam(lr=0.0001)

    gan.compile(optimizer=gan_optimizer,
                loss=['mae', 'mse', 'mse', 'binary_crossentropy', 'mse', 'mse', 'mse'],
                loss_weights=[1, 100, 100, gan_weight, 0.001, 0.001, 0.001]
                )

    print(gan.metrics_names)

    # csv_logger = CSVLogger('log_%s.csv' % started_time_date, append=True, separator=';')
    f = None
    wr = None

    folder_name = "%s_%s_%s" % (model.name, data.name, started_time_date)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    started_time = time.time()
    print_per = 100
    for i in range(max_iter):
        sampled_input_images, sampled_input_views, sampled_target_images = data.get_batched_data(batch_size,
                                                                                                 resize_shape=model.image_size,
                                                                                                 single_model=single_model)
        sampled_input_images = model.pixel_normalizer(sampled_input_images)
        sampled_target_images = model.pixel_normalizer(sampled_target_images)
        # f.write("%d\n"%i)
        # f.flush()

        generated_images = model.get_model().predict(
            {'image_input': sampled_input_images, 'viewpoint_input': sampled_input_views}
        )

        real_images = sampled_target_images
        combined_images = np.concatenate([generated_images, real_images])
        labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
        # labels += 0.05 * np.random.random(labels.shape)

        d_loss = [0, 0]
        if i % 2 == 0:
            d_loss = discriminator.get_model().train_on_batch(combined_images, labels)

        misleading_targets = np.zeros((batch_size, 1))
        (d_1, d_2, d_o) = discriminator.get_model().predict(sampled_target_images)
        (v_1, v_2, v_3) = vgg_model.get_model().predict(sampled_target_images)

        g_loss = gan.train_on_batch(
            [sampled_input_images, sampled_input_views],
            [sampled_target_images, d_1, d_2, misleading_targets, v_1, v_2, v_3],
            #
            # {'image_input': sampled_input_images, 'viewpoint_input': sampled_input_views},
            # {'main_output': sampled_target_images, 'gan_output': [d_1, d_2, misleading_targets], 'model_6':[v_1, v_2, v_3]}
        )

        print("D loss", d_loss)
        print(gan.metrics_names)
        print("G_loss", g_loss)

        if i % print_per == 0:
            print(i)
            elapsed_time = time.time() - started_time
            print("time elapsed from first %s" % time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

            if write_log:
                if wr is None:
                    import csv
                    f = open('%s/log_%s.csv' % (folder_name, started_time_date), 'w', encoding='utf-8')
                    wr = csv.writer(f)
                    wr.writerow(
                        ["epoch"] + discriminator.get_model().metrics_names + gan.metrics_names + ["elapsed_time"])

                    # f.write("%s\t%s\n" % ("D_loss", '\t'.join(gan.metrics_names)))
                wr.writerow([i] + d_loss + g_loss + [time.strftime("%H:%M:%S", time.gmtime(elapsed_time))])
                # f.write("%f\t%s\n" % (d_loss[1], '\t'.join([str(x) for x in a_loss])))
                f.flush()

        if i % export_image_per == 0:
            test_few_models_and_export_image(model, data, i, folder_name, test_n=5, single_model=False)

    if save_model:
        model.save_model(data.name, started_time_date)
        discriminator.save_model(data.name, started_time_date)


def train_with_mutiple_discriminator(model: ModelInterface,
                                     data: DataLoader,
                                     max_iter=100000,
                                     save_model=True,
                                     write_log=True,
                                     gan_weight=1,
                                     export_image_per=1000,
                                     single_model=False,
                                     batch_size=32):
    from keras.optimizers import Adam

    discriminator = Discriminator(model.image_size)
    discriminator.build_model()
    discriminator.get_model().compile(optimizer=Adam(lr=0.0001), loss={'gan_loss': 'binary_crossentropy'})
    discriminator.get_model().trainable = False

    azimuth_discriminator = Discriminator(model.image_size, output_size=data.n_azimuth, name="azimuth_discriminator")
    azimuth_discriminator.build_model()
    azimuth_discriminator.get_model().compile(optimizer=Adam(lr=0.0001), loss={'gan_loss': keras.losses.categorical_crossentropy})
    azimuth_discriminator.get_model().trainable = False

    elevation_discriminator = Discriminator(model.image_size, output_size=data.n_elevation, name="elevation_discriminator")
    elevation_discriminator.build_model()
    elevation_discriminator.get_model().compile(optimizer=Adam(lr=0.0001), loss={'gan_loss': keras.losses.categorical_crossentropy})
    elevation_discriminator.get_model().trainable = False

    model_identity_discriminator = IdentityDiscriminator(model.image_size)
    model_identity_discriminator.build_model()
    model_identity_discriminator.get_model().compile(optimizer=Adam(lr=0.0001), loss={'gan_loss': 'binary_crossentropy'})
    model_identity_discriminator.get_model().trainable = False

    model.get_model().name = "generator"
    image_input = keras.Input(shape=(model.image_size, model.image_size, 3), name='gan_image_input')
    viewpoint_input = keras.Input(shape=(42,), name='gan_viewpoint_input')

    generated_image = model.get_model()([image_input, viewpoint_input])

    gan_output = discriminator.get_model()(generated_image)
    azimuth_output = azimuth_discriminator.get_model()(generated_image)
    elevation_output = elevation_discriminator.get_model()(generated_image)
    identity_output = model_identity_discriminator.get_model()([image_input, generated_image])

    vgg_model = VGG16Model(model.image_size)
    vgg_model.build_model()
    vgg_feature_output = vgg_model.get_model()(generated_image)

    gan = Model(inputs=[image_input, viewpoint_input], outputs=[generated_image] + gan_output + vgg_feature_output + [azimuth_output[2], elevation_output[2], identity_output])
    # vgg_model.get_model().get_layer('block1_pool').output,
    # vgg_model.get_model().get_layer('block2_pool').output,
    # vgg_model.get_model().get_layer('block3_pool').output,
    # discriminator.get_model().get_layer('d_en_conv_0').output,
    # discriminator.get_model().get_layer('d_en_conv_1').output,
    # discriminator.get_model().get_layer('gan_loss').output
    # ])
    gan.summary()

    # visualize_model(gan)
    started_time_date = time.strftime("%Y%m%d_%H%M%S")

    gan_optimizer = Adam(lr=0.0001)

    gan.compile(optimizer=gan_optimizer,
                loss=['mae', 'mse', 'mse', 'binary_crossentropy', 'mse', 'mse', 'mse', 'categorical_crossentropy', 'categorical_crossentropy', 'binary_crossentropy'],
                loss_weights=[1.0, 100.0, 100.0, gan_weight, 0.001, 0.001, 0.001, 1.0, 1.0, 1.0]
                )
    print(gan.metrics_names)

    # csv_logger = CSVLogger('log_%s.csv' % started_time_date, append=True, separator=';')
    f = None
    wr = None

    folder_name = "%s_%s_%s" % (model.name, data.name, started_time_date)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    started_time = time.time()
    print_per = 100
    for i in range(max_iter):
        sampled_input_images, sampled_input_views, sampled_target_images, azielinfo = data.get_batched_data(batch_size,
                                                                                                 resize_shape=model.image_size,
                                                                                                 single_model=single_model,
                                                                                                            not_concatenate=True)
        sampled_input_images = model.pixel_normalizer(sampled_input_images)
        sampled_target_images = model.pixel_normalizer(sampled_target_images)
        # f.write("%d\n"%i)
        # f.flush()

        generated_images = model.get_model().predict(
            {'image_input': sampled_input_images, 'viewpoint_input': sampled_input_views}
        )

        combined_images = np.concatenate([generated_images, sampled_target_images])
        combined_input_images = np.concatenate([sampled_input_images, sampled_input_images])

        labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
        azimuth_labels = np.concatenate([np.ones((batch_size, data.n_azimuth)) / data.n_azimuth, azielinfo[3]])
        elevation_labels = np.concatenate([np.ones((batch_size, data.n_elevation)) / data.n_elevation, azielinfo[2]])

        # labels += 0.05 * np.random.random(labels.shape)

        d_loss = [0, 0]
        if i % 2 == 0:
            d_loss = discriminator.get_model().train_on_batch(combined_images, labels)
            az_loss = azimuth_discriminator.get_model().train_on_batch(combined_images, azimuth_labels)
            el_loss = elevation_discriminator.get_model().train_on_batch(combined_images, elevation_labels)
            identity_loss = model_identity_discriminator.get_model().train_on_batch([combined_images, combined_input_images], labels)


        misleading_targets = np.zeros((batch_size, 1))
        (d_1, d_2, d_o) = discriminator.get_model().predict(sampled_target_images)
        (v_1, v_2, v_3) = vgg_model.get_model().predict(sampled_target_images)

        g_loss = gan.train_on_batch(
            [sampled_input_images, sampled_input_views],
            [sampled_target_images, d_1, d_2, misleading_targets, v_1, v_2, v_3, azielinfo[3], azielinfo[2], misleading_targets]
        )

        print("D loss", d_loss)
        print(gan.metrics_names)
        print("G_loss", g_loss)

        if i % print_per == 0:
            print(i)
            elapsed_time = time.time() - started_time
            print("time elapsed from first %s" % time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

            if write_log:
                if wr is None:
                    import csv
                    f = open('%s/log_%s.csv' % (folder_name, started_time_date), 'w', encoding='utf-8')
                    wr = csv.writer(f)
                    wr.writerow(
                        ["epoch"] + discriminator.get_model().metrics_names + gan.metrics_names + ["elapsed_time"])

                    # f.write("%s\t%s\n" % ("D_loss", '\t'.join(gan.metrics_names)))
                wr.writerow([i] + d_loss + g_loss + [time.strftime("%H:%M:%S", time.gmtime(elapsed_time))])
                # f.write("%f\t%s\n" % (d_loss[1], '\t'.join([str(x) for x in a_loss])))
                f.flush()

        if i % export_image_per == 0:
            test_few_models_and_export_image(model, data, i, folder_name, test_n=5, single_model=False)

    if save_model:
        model.save_model(data.name, started_time_date)
        discriminator.save_model(data.name, started_time_date)




def train_afp(model: ModelInterface, data: DataLoader, max_iter=100000,
              lr=0.0001,
              load_model_numbers=2048,
              print_log=True, save_model=True):
    started_time_date = time.strftime("%Y%m%d_%H%M%S")
    from keras.optimizers import Adam
    from keras import losses

    optimizer = Adam(lr=lr, beta_1=0.9)
    model.get_model().compile(optimizer=optimizer,
                              loss={'main_output': losses.mean_absolute_error, 'mask_output': 'binary_crossentropy'},
                              loss_weights={'main_output': 1, 'mask_output': 0.1})

    csv_logger = CSVLogger('log_%s.csv' % started_time_date, append=True, separator=';')
    batch_size = 32
    started_time = time.time()
    print_per = 100

    for i in range(max_iter):
        batch_started_time = time.time()
        input_image, view_point, target_image = data.get_batched_data(load_model_numbers)
        batching_elapsed_time = time.time() - batch_started_time
        # print("batching time %s" % time.strftime("%H:%M:%S", time.gmtime(batching_elapsed_time)))

        target_image_mask = get_background(target_image)
        input_image = model.pixel_normalizer(input_image)
        target_image = model.pixel_normalizer(target_image)

        # f.write("%d\n"%i)
        # f.flush()

        if print_log:
            model.get_model().fit({'image_input': input_image, 'viewpoint_input': view_point},
                                  {'main_output': target_image, 'mask_output': target_image_mask},
                                  batch_size=batch_size, callbacks=[csv_logger],
                                  verbose=True)  # if i% print_per !=0 else True)
        else:
            model.get_model().fit({'image_input': input_image, 'viewpoint_input': view_point},
                                  {'main_output': target_image, 'mask_output': target_image_mask},
                                  batch_size=batch_size, verbose=True)  # if i% print_per !=0 else True)
        print(i)
        elapsed_time = time.time() - started_time
        print("time elapsed from first %s" % time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    elapsed_time = time.time() - started_time
    print("time elapsed from first %s" % time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # model.get_model().save_weights("model_%s.h5"%started_time_date)
    if save_model:
        model.save_model(data.name, started_time_date)


index = 0


def train_afp_faster(model: ModelInterface,
                     data: DataGenerator,
                     max_iter=100000,
                     batch_size=32,
                     lr=0.0001,
                     workers=6,
                     print_log=True,
                     save_model=True):
    started_time_date = time.strftime("%Y%m%d_%H%M%S")
    started_time = time.time()
    from keras.optimizers import Adam
    from keras import losses
    csv_logger = CSVLogger('log_%s.csv' % started_time_date, append=True, separator=';')

    optimizer = Adam(lr=lr, beta_1=0.9)
    model.get_model().compile(optimizer=optimizer,
                              loss={'main_output': losses.mean_absolute_error, 'mask_output': 'binary_crossentropy'},
                              loss_weights={'main_output': 1, 'mask_output': 0.01})

    f = open('log_%s.txt' % started_time_date, 'w')
    data.max_iter = max_iter
    data.batch_size = batch_size

    def batchOutput(batch, logs):
        if batch % 100 == 0:
            f.write('%s\n' % logs)
            f.flush()

    from keras.callbacks import LambdaCallback
    batchLogCallback = LambdaCallback(on_batch_end=batchOutput)

    if print_log:
        callbacks = [batchLogCallback, csv_logger]
    else:
        callbacks = []

    model.get_model().fit_generator(
        generator=data,
        use_multiprocessing=True,
        workers=workers,
        callbacks=callbacks
    )

    elapsed_time = time.time() - started_time
    print("time elapsed from first %s" % time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    if save_model:
        model.save_model(data.data_container.name, started_time_date)


def train_faster(model: ModelInterface,
                 data: DataGenerator,
                 max_iter=100000,
                 batch_size=32,
                 lr=0.0001,
                 workers=6,
                 print_log=True,
                 save_model=True,
                 single_model=False,
                 save_images_per=5000):
    data.single_model = single_model
    started_time_date = time.strftime("%Y%m%d_%H%M%S")
    started_time = time.time()
    from keras.optimizers import Adam

    optimizer = Adam(lr=lr)
    model.get_model().compile(optimizer=optimizer, loss={'main_output': 'mean_absolute_error'})
    folder_name = "%s_%s_%s" % (model.name, data.data_container.name, started_time_date)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    f = open('%s/log_%s.txt' % (folder_name, started_time_date), 'w')

    data.max_iter = max_iter
    data.batch_size = batch_size

    def batch_output(batch, logs):
        if batch % 100 == 0:
            elapsed_time = time.time() - started_time
            f.write('%s\t%s\n' % (logs, time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
            f.flush()

        if save_images_per != -1 and batch % save_images_per == 0:
            if batch == 0:
                data.data_container.load_h5_data()
            test_few_models_and_export_image(model, data.data_container, batch, folder_name, test_n=5,
                                             single_model=False)

    from keras.callbacks import LambdaCallback
    batchLogCallback = LambdaCallback(on_batch_end=batch_output)

    if print_log:
        callbacks = [batchLogCallback]
    else:
        callbacks = []

    model.get_model().fit_generator(
        generator=data,
        use_multiprocessing=True,
        workers=workers,
        callbacks=callbacks
    )

    if save_model:
        model.save_model(data.data_container.name, started_time_date)


def train(model: ModelInterface,
          data: DataLoader,
          max_iter=100000,
          batch_size=32,
          lr=0.0001,
          workers=6,
          print_log=True,
          save_model=True,
          single_model=False,
          save_images_per=5000):
    started_time_date = time.strftime("%Y%m%d_%H%M%S")
    started_time = time.time()
    from keras.optimizers import Adam

    optimizer = Adam(lr=lr)
    model.get_model().compile(optimizer=optimizer,
                              loss={'main_output': 'mean_absolute_error'})
    folder_name = "%s_%s_%s" % (model.name, data.name, started_time_date)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    f = open('%s/log_%s.txt' % (folder_name, started_time_date), 'w')

    data.max_iter = max_iter
    data.batch_size = batch_size

    print_per = 100
    for i in range(max_iter):
        sampled_data_0, sampled_data_1, sampled_data_2 = data.get_batched_data(batch_size, single_model=single_model)
        sampled_data_0 = model.pixel_normalizer(sampled_data_0)
        sampled_data_2 = model.pixel_normalizer(sampled_data_2)

        if i % print_per == 0:
            elapsed_time = time.time() - started_time
            f.write('%s\t%s\n' % (i, time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
            f.flush()

        if save_images_per != -1 and i % save_images_per == 0:
            if i == 0:
                data.load_h5_data()
            test_few_models_and_export_image(model, data, i, folder_name, test_n=5,
                                             single_model=False)

        model.get_model().fit({'image_input': sampled_data_0, 'viewpoint_input': sampled_data_1},
                              {'main_output': sampled_data_2},
                              batch_size=batch_size,
                              verbose=False)

    if save_model:
        model.save_model('%s/%s' % (folder_name, data.name), started_time_date)

    # started_time_date = time.strftime("%Y%m%d_%H%M%S")
    # from keras.optimizers import Adam
    # optimizer = Adam(lr=0.0001, beta_1=0.9)
    #
    # model.get_model().compile(optimizer=optimizer, loss='mse')
    # csv_logger = CSVLogger('log_%s.csv' % started_time_date, append=True, separator=';')
    # batch_size = 32
    # started_time = time.time()
    # print_per = 100
    # for i in range(max_iter):
    #     sampled_data_0, sampled_data_1, sampled_data_2 = data.get_batched_data(batch_size)
    #     sampled_data_0 = model.pixel_normalizer(sampled_data_0)
    #     sampled_data_2 = model.pixel_normalizer(sampled_data_2)
    #
    #     if i % print_per == 0:
    #         print(i)
    #         elapsed_time = time.time() - started_time
    #         f.write('%s\n' % (time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
    #
    #     model.get_model().fit({'image_input': sampled_data_0, 'viewpoint_input': sampled_data_1},
    #               {'main_output': sampled_data_2},
    #               batch_size=batch_size, callbacks=[csv_logger],
    #               verbose=False)
    # model.save_model(data.name, started_time_date)
    #


def train_sequential_old(model: ModelSequential,
                         data: DataGenerator,
                         max_iter=100000,
                         batch_size=32,
                         lr=0.0001,
                         workers=6,
                         print_log=True,
                         save_model=True):
    started_time_date = time.strftime("%Y%m%d_%H%M%S")
    started_time = time.time()
    from keras.optimizers import Adam
    from keras import losses
    csv_logger = CSVLogger('log_%s.csv' % started_time_date, append=True, separator=';')

    optimizer = Adam(lr=lr, beta_1=0.9)
    model.get_model().compile(optimizer=optimizer,
                              loss={'main_output': losses.mean_squared_error})

    optimizer2 = Adam(lr=lr, beta_1=0.9)
    model.afn.get_model().compile(optimizer=optimizer2,
                                  loss={'main_output': losses.mean_absolute_error,
                                        'mask_output': 'binary_crossentropy'},
                                  loss_weights={'main_output': 1, 'mask_output': 0.01})

    f = open('log_%s.txt' % started_time_date, 'w')
    data.max_iter = max_iter
    data.batch_size = batch_size

    def batchOutput(batch, logs):
        if batch % 100 == 0:
            f.write('%s\n' % logs)
            f.flush()

    from keras.callbacks import LambdaCallback
    batchLogCallback = LambdaCallback(on_batch_end=batchOutput)

    if print_log:
        callbacks = [batchLogCallback, csv_logger]
    else:
        callbacks = []

    model.get_model().fit_generator(
        generator=data,
        use_multiprocessing=True,
        workers=workers,
        callbacks=callbacks
    )

    elapsed_time = time.time() - started_time
    print("time elapsed from first %s" % time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    if save_model:
        model.save_model(data.data_container.name, started_time_date)


def train_sequential(model: ModelSequential,
                     afn: ModelAppearanceFlow,
                     data: DataLoader,
                     max_iter=100000,
                     batch_size=32,
                     lr=0.0001,
                     workers=6,
                     print_log=True,
                     save_model=True):
    started_time_date = time.strftime("%Y%m%d_%H%M%S")
    from keras.optimizers import Adam
    optimizer = Adam(lr=lr, beta_1=0.9)

    model.get_model().compile(optimizer=optimizer, loss='mse')
    csv_logger = CSVLogger('log_%s.csv' % started_time_date, append=True, separator=';')
    batch_size = 32
    started_time = time.time()
    print_per = 100

    intermediate_layer_model = Model(inputs=afn.get_model().input,
                                     outputs=afn.get_model().get_layer('image_feature').output)

    if print_log:
        callbacks = [csv_logger]
    else:
        callbacks = []

    for i in range(max_iter):
        sampled_data_0, sampled_data_1, sampled_data_2 = data.get_batched_data(batch_size)
        sampled_data_0 = model.pixel_normalizer(sampled_data_0)
        sampled_data_2 = model.pixel_normalizer(sampled_data_2)

        (pred_image, pred_mask) = afn.get_model().predict(
            {'image_input': sampled_data_0, 'viewpoint_input': sampled_data_1})
        image_feature = intermediate_layer_model.predict(
            {'image_input': sampled_data_0, 'viewpoint_input': sampled_data_1})
        # f.write("%d\n"%i)
        # f.flush()
        if i % print_per == 0:
            print(i)
            elapsed_time = time.time() - started_time
            print("time elapsed from first %s" % time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

        model.get_model().fit(
            {'image_input': pred_image, 'viewpoint_input': sampled_data_1, 'image_feature_input': image_feature},
            {'tanh_out': sampled_data_2},
            batch_size=batch_size,
            verbose=True,
            callbacks=callbacks)  # if i% print_per !=0 else True)

    # model.get_model().save_weights("model_%s.h5"%started_time_date)
    model.save_model(data.name, started_time_date)


from typing import List


def test_for_all_models(data: DataLoader, model_list: List[ModelInterface]):
    absolute_errors = np.zeros((len(model_list), len(data.model_list)))
    ssim_errors = np.zeros((len(model_list), len(data.model_list)))

    for k, model_name in enumerate(data.model_list):

        for m in range(18):
            input_image_original, target_image_original, pose_info = data.get_batched_data_single(start_angle=m, model_name=model_name)

        input_image_original, target_image_original, pose_info = data.get_batched_data(32, single_model=True,
                                                                                        model_name=model_name)
        for i, model in enumerate(model_list):
            # metrics = ms[i].evaluate([input_image_original, pose_info], target_image_original, verbose=False)
            pose_info_per_model = model.process_pose_info(data, pose_info)
            metrics = model.evaluate(input_image_original, target_image_original, pose_info_per_model)

            # pred_images = model.get_predicted_image((input_image_original, view_point))
            # error = K.eval(K.mean(K.abs(pred_images - target_image_original)))
            # # ssim_value = K.eval(K.mean(tf.image.ssim(tf.convert_to_tensor(target_image_original, dtype=tf.float32),
            # #                                   tf.convert_to_tensor(pred_images, dtype=tf.float32),
            # #                                   max_val=1.0)))
            # ssim_value = K.eval(K.mean(tf.image.ssim(target_image_original, pred_images, max_val=1.0)))
            absolute_errors[i][k] = metrics[1]
            ssim_errors[i][k] = metrics[2]

        #print(k, absolute_errors[:,k], ssim_errors[:,k])

    print("MAE", np.mean(absolute_errors, axis=1))
    print("SSIM", np.mean(ssim_errors, axis=1))


def test_for_all_models_thorough(data: DataLoader, model_list: List[ModelInterface], export_file=None):
    absolute_errors = np.zeros((len(model_list), len(data.model_list), 18))
    ssim_errors = np.zeros((len(model_list), len(data.model_list), 18))

    for k, model_name in enumerate(data.model_list):
        print(model_name)
        for m in range(18):
            input_image_original, target_image_original, pose_info = data.get_batched_data_single(start_angle=m, model_name=model_name)

            for i, model in enumerate(model_list):
                # metrics = ms[i].evaluate([input_image_original, pose_info], target_image_original, verbose=False)
                pose_info_per_model = model.process_pose_info(data, pose_info)
                metrics = model.evaluate(input_image_original, target_image_original, pose_info_per_model)

                # pred_images = model.get_predicted_image((input_image_original, view_point))
                # error = K.eval(K.mean(K.abs(pred_images - target_image_original)))
                # # ssim_value = K.eval(K.mean(tf.image.ssim(tf.convert_to_tensor(target_image_original, dtype=tf.float32),
                # #                                   tf.convert_to_tensor(pred_images, dtype=tf.float32),
                # #                                   max_val=1.0)))
                # ssim_value = K.eval(K.mean(tf.image.ssim(target_image_original, pred_images, max_val=1.0)))

                absolute_errors[i][k][m] = metrics[1]
                ssim_errors[i][k][m] = metrics[2]
        # print(k, absolute_errors[:,k], ssim_errors[:,k])

    mae = np.mean(absolute_errors, axis=(1, 2))
    ssim = np.mean(ssim_errors, axis=(1, 2))
    raw_data = {'name': [model.name for model in model_list],
                'col1': mae,
                'col2': ssim}
    df = DataFrame(raw_data)
    df = df.set_index("name")

    print("MAE", mae)
    print("SSIM", ssim)

    if export_file is not None:
        df.to_csv("%s.csv" % export_file)


# def test_for_all_models_thorough_per_model(data: DataContainer, model):
#     absolute_errors = np.zeros((len(data.model_list), 18))
#     ssim_errors = np.zeros((len(data.model_list), 18))
#
#     for k, model_name in enumerate(data.model_list):
#         print(model_name)
#         for m in range(18):
#             input_image_original, target_image_original, pose_info = data.get_batched_data_single(start_angle=m, model_name=model_name)
#
#             # metrics = ms[i].evaluate([input_image_original, pose_info], target_image_original, verbose=False)
#             pose_info_per_model = model.process_pose_info(data, pose_info)
#             metrics = model.evaluate(input_image_original, target_image_original, pose_info_per_model)
#
#             absolute_errors[k][m] = metrics[1]
#             ssim_errors[k][m] = metrics[2]
#
#     mae = np.mean(absolute_errors)
#     ssim = np.mean(ssim_errors)
#
#     return mae, ssim

def test_for_error_calc(data: DataLoader, model: ModelInterface, iter=100, batch_size=32):
    for i in range(iter):
        sampled_data = data.get_batched_data(batch_size, single_model=True)
        result = model.get_model().predict(
            {'image_input': model.pixel_normalizer(sampled_data[0]), 'viewpoint_input': sampled_data[1]})
        error = K.eval(K.mean(keras.losses.mean_squared_error(sampled_data[2], model.pixel_normalizer_reverse(result))))
        # print(np.array(error).shape)
        print(error)


def test_for_few_samples(data: DataLoader, model: ModelInterface, test_n=5):
    f, axarr = plt.subplots(test_n, 3, figsize=(24, 24))
    input_image_original, view_point, target_image_original = data.get_batched_data(test_n, single_model=False)
    result = model.get_predicted_image((input_image_original, view_point))
    # result = model.get_model().predict({'image_input': model.pixel_normalizer(sampled_data[0]),
    #                                     'viewpoint_input': sampled_data[1]})
    # print(result[0][128][128:138])
    for i in range(test_n):
        axarr[i, 0].imshow(input_image_original[i])
        axarr[i, 1].imshow(result[i])
        axarr[i, 2].imshow(target_image_original[i])


def test_for_few_samples_afn(data: DataLoader, model: ModelInterface, test_n=5):
    f, axarr = plt.subplots(test_n, 5, figsize=(24, 24))
    # sampled_data = data.get_batched_data(test_n, single_model=False)

    input_image_original, view_point, target_image_original = data.get_batched_data(test_n, single_model=False)
    # target_image_mask = get_background(target_image_original)
    input_image = model.pixel_normalizer(input_image_original)
    target_image = model.pixel_normalizer(target_image_original)

    (pred_image, pred_mask) = model.get_model().predict({'image_input': input_image, 'viewpoint_input': view_point})

    intermediate_layer_model = Model(inputs=model.get_model().input,
                                     outputs=model.get_model().get_layer("conv2d_transpose_6").output)
    intermediate_output = intermediate_layer_model.predict({'image_input': input_image, 'viewpoint_input': view_point})

    # print(intermediate_output[0][0][128:256])

    pred_image = model.pixel_normalizer_reverse(pred_image)
    # print(pred_image[0][0][128:138])
    # print(result[0][128][128:138])
    for i in range(test_n):
        axarr[i, 0].imshow(input_image_original[i])
        axarr[i, 1].imshow(target_image_original[i])
        axarr[i, 2].imshow(pred_image[i])
        m = pred_mask[i][..., 0]
        t = np.where(m > 0.5, 1.0, 0.0)
        # t = np.argmax(m, axis = 2)
        # print(t[0])
        R = np.stack((t, t, t), axis=2)
        axarr[i, 3].imshow(R)
        # axarr[i, 3].imshow(pred_image[i] * R + (1-R))
        dim = np.zeros((256, 256))
        R2 = np.stack((intermediate_output[i][..., 0], dim, intermediate_output[i][..., 1]), axis=2)
        axarr[i, 4].imshow(R2)

from PIL import Image
def do_comparison(data: DataLoader, model_list: List[ModelInterface], test_n=5, model_id=None, model_info=None):
    print(model_list[0].image_size)
    if model_info is not None:
        lst1, lst2, lst3 = zip(*model_info)
        test_n = len(lst1)
        input_image_original, target_image_original, poseinfo = data.get_batched_data_from_info(lst1,lst2,lst3)
    else:
        if model_id is not None:
            input_image_original, target_image_original, poseinfo = data.get_batched_data(test_n, single_model=True, model_name=model_id, verbose=True)
        else:
            input_image_original, target_image_original, poseinfo = data.get_batched_data(test_n, single_model=False, verbose=True)


    pred_images_list = []
    errors = []
    y_true = K.variable(target_image_original)

    image_size = data.image_size
    N = len(model_list)
    export_image = np.zeros((image_size * test_n, image_size * (N + 2), 3), dtype=np.float32)

    #tf.enable_eager_execution()
    for model in model_list:
        per_model_poseinfo = model.process_pose_info(data, poseinfo)
        pred_images = model.get_predicted_image((input_image_original, per_model_poseinfo))
        pred_images_list.append(pred_images)
        y_pred = K.variable(pred_images)
        error = K.eval(K.mean(K.abs(y_pred - y_true), axis=[1, 2, 3]))
        errors.append(error)

    f, axarr = plt.subplots(test_n, len(model_list) + 2, figsize=(24, 24 * test_n / 5))

    for i in range(test_n):
        axarr[i, 0].imshow(input_image_original[i])
        axarr[i, 1].imshow(target_image_original[i])
        axarr[i, 0].axis('off')
        axarr[i, 1].axis('off')
        #axarr[i, 0].set_title("Input Image")
        #axarr[i, 1].set_title("Target Image")
        put_image_in(export_image, input_image_original[i], i, 0, image_size)
        put_image_in(export_image, target_image_original[i], i, 1, image_size)

        for j, model in enumerate(model_list):
            axarr[i, j + 2].imshow(pred_images_list[j][i])

            put_image_in(export_image, pred_images_list[j][i], i, j+2, image_size)

            ssim_value = K.eval(tf.image.ssim(tf.convert_to_tensor(target_image_original[i], dtype=tf.float32),
                                              tf.convert_to_tensor(pred_images_list[j][i], dtype=tf.float32),
                                              max_val=1.0))
            axarr[i, j + 2].set_title("L1 loss : %.4f\nSSIM : %.4f" % (errors[j][i], ssim_value))
            axarr[i, j + 2].axis('off')

    f.tight_layout()
    plt.show()
    return export_image

def save_all_images(data:DataLoader):
    from PIL import Image
    for i in range(len(data.model_list)):
        image = data.all_images_for_all_model[i][2]
        print(image.shape)
        result = Image.fromarray((image * 255).astype(np.uint8))
        result.save('test_images_examples_chair/%d.png' % i)

def do_comparison_export(data: DataLoader, model_list: List[ModelInterface], model_id=None):
    print(model_list[0].image_size)
    input_image_original, target_image_original, poseinfo = data.get_batched_data(1, model_name=model_id)
    pred_images_list = []
    errors = []
    y_true = K.variable(target_image_original)

    # tf.enable_eager_execution()
    for model in model_list:
        per_model_poseinfo = model.process_pose_info(data, poseinfo)
        pred_images = model.get_predicted_image((input_image_original, per_model_poseinfo))
        pred_images_list.append(pred_images)
        y_pred = K.variable(pred_images)
        error = K.eval(K.mean(K.abs(y_pred - y_true), axis=[1, 2, 3]))
        errors.append(error)

    A = 2
    B = 8
    output_image = np.zeros((64 * A, 64 * (B + 2), 3))
    pointer = 0
    for i in range(A):
        sx = i * 64
        sy = 0
        output_image[sx:sx + 64, sy:sy + 64, :] = input_image_original[0]
        sx = i * 64
        sy = 64
        output_image[sx:sx + 64, sy:sy + 64, :] = target_image_original[0]

        for j in range(B):
            C = pred_images_list[pointer]
            print(C.shape)
            sx = i*64
            sy = (j+2)*64
            output_image[sx:sx+64,sy:sy+64,:] = pred_images_list[pointer][0,:,:,:]
            pointer = pointer + 1
    return output_image
    # import scipy.misc
    # scipy.misc.imsave('outfile.jpg', output_image)

    #
    # N = len(model_list) // 2
    # f, axarr = plt.subplots(2, N + 2, figsize=(24, 24))
    #
    # for i in range(2):
    #     axarr[i, 0].imshow(input_image_original[i])
    #     axarr[i, 1].imshow(target_image_original[i])
    #     axarr[i, 0].axis('off')
    #     axarr[i, 1].axis('off')
    #     # axarr[i, 0].set_title("Input Image")
    #     # axarr[i, 1].set_title("Target Image")
    #
    #     for j, model in enumerate(model_list):
    #         axarr[i, j + 2].imshow(pred_images_list[j][i])
    #         ssim_value = K.eval(tf.image.ssim(tf.convert_to_tensor(target_image_original[i], dtype=tf.float32),
    #                                           tf.convert_to_tensor(pred_images_list[j][i], dtype=tf.float32),
    #                                           max_val=1.0))
    #         axarr[i, j + 2].set_title("L1 loss : %.4f\nSSIM : %.4f" % (errors[j][i], ssim_value))
    #         axarr[i, j + 2].axis('off')
    #
    # f.tight_layout()
    # plt.show()

def show_attention_map(data: DataLoader, model_list: List[ModelInterface], test_n=5):
    print(model_list[0].image_size)
    input_image_original, target_image_original, poseinfo = data.get_batched_data(test_n, single_model=False)
    pred_images_list = []
    errors = []
    y_true = K.variable(target_image_original)
    attention_maps = {}
    feature_maps = {}

    s = K.get_session()
    #tf.enable_eager_execution()
    for model in model_list:
        per_model_poseinfo = model.process_pose_info(data, poseinfo)
        pred_images = model.get_predicted_image((input_image_original, per_model_poseinfo))

        for q, layer in model.custom_attn_layers.items():
            attn_map, input_h = s.run([layer.beta, layer.input_h], feed_dict={model.model.input[0]: input_image_original, model.model.input[1]: per_model_poseinfo})
            # attn_map, input_h = s.run([layer.gamma],
            #                           feed_dict={model.model.input[0]: input_image_original,
            #                                      model.model.input[1]: per_model_poseinfo})
            #gamma = s.run(layer.gamma)

            if q not in attention_maps:
                attention_maps[q] = []
                feature_maps[q] = []

            attention_maps[q].append(attn_map)
            feature_maps[q].append(input_h)

        pred_images_list.append(pred_images)
        y_pred = K.variable(pred_images)
        error = K.eval(K.mean(K.abs(y_pred - y_true), axis=[1, 2, 3]))
        errors.append(error)
    ks = [8, 16, 32]

    positions = []
    for i in range(test_n):
        while True:
            tx = np.random.random(1)
            ty = np.random.random(1)
            q = 64

            rx = int(tx * q)
            ry = int(ty * q)

            if np.sum(input_image_original[i][ry][rx]) > 2.7 or np.sum(target_image_original[i][ry][rx]) > 2.7:
                continue

            input_image_original[i][ry][rx][0] = 1
            input_image_original[i][ry][rx][1] = 0
            input_image_original[i][ry][rx][2] = 0
            target_image_original[i][ry][rx][0] = 1
            target_image_original[i][ry][rx][1] = 0
            target_image_original[i][ry][rx][2] = 0
            positions.append((tx, ty))
            break

    for q in ks:
        f, axarr = plt.subplots(test_n, len(model_list)+2, figsize=(24, 24 * test_n / 5))

        for i in range(test_n):
            tx, ty = positions[i]
            rx = int(tx * q)
            ry = int(ty * q)
            rxy = ry * q + rx

            axarr[i, 0].imshow(input_image_original[i])
            axarr[i, 1].imshow(target_image_original[i])
            axarr[i, 0].axis('off')
            axarr[i, 1].axis('off')

            for j, model in enumerate(model_list):
                A = np.reshape(attention_maps[q][j][i][rxy], (q, q))
                #A = np.stack((A,A,A), axis=2)
                m = np.amin(A)
                ma = np.amax(A)
                #A = (A - m) / (ma - m)
                #print(A.shape)
                axarr[i, j + 2].imshow(A)
                axarr[i, j + 2].axis('off')
                #axarr[i, j + 2].imshow(np.reshape(feature_maps[q][j][i][:,:,0], (q, q)))

                #print(np.sum(attention_maps[q][j][i][t]))
        f.tight_layout()
        plt.show()




    # for j, model in enumerate(model_list):
    #    imageio.mimwrite('%s_%s_output_filename.mp4'% (data.name, model.name), videos[j], fps=2)

    # (pred_image, pred_mask) = model.get_model().predict({'image_input': input_image, 'viewpoint_input': view_point})

from olds.model.model_angle_classifier import AzimuthClassifier


def test_with_classifier(data: DataLoader, test_data: DataLoader, batch_size = 32, max_iter = 1000, date_type='azimuth'):
    model = AzimuthClassifier(data.image_size, data.n_azimuth)
    model.build_model()
    model.get_model().compile(optimizer=Adam(lr=0.001), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    data_generator = DataGenerator(model, data, data_type=date_type)
    data_generator.max_iter = max_iter

    def test_few(batch, logs):
        if batch % 1000 == 0:

            if test_data is not None:
                x, y = test_data.get_batched_data_labeled(batch_size=100)
                score = model.get_model().evaluate(x, y, verbose=0)
                print('Test loss:', score[0])
                print('Test accuracy:', score[1])

            x, y = data.get_batched_data_labeled(batch_size=100)
            score = model.get_model().evaluate(x, y, verbose=0)
            print('Train loss:', score[0])
            print('Train accuracy:', score[1])

    from keras.callbacks import LambdaCallback
    batch_callback = LambdaCallback(on_batch_end=test_few)
    model.get_model().fit_generator(data_generator, verbose=False, callbacks=[batch_callback], use_multiprocessing=False)

    eval_data_generator = DataGenerator(model, test_data if test_data is not None else data, data_type=date_type)
    eval_data_generator.max_iter = 10000
    score = model.get_model().evaluate_generator(eval_data_generator, use_multiprocessing=False)
    print('Final test loss:', score[0])
    print('Final test accuracy:', score[1])


# from model.model_encoder import ModelDisentangleModel
# def train_disentangle_model(data: DataContainer, decoder_type = 'pg', max_iter = 200000, batch_size=32, export_image_per = 2000, save_model=True):
#     model = ModelDisentangleModel(image_size = data.image_size, decoder_type=decoder_type)
#     model.build_model()
#     model.get_model().compile(optimizer=Adam(0.0001), loss=['mae', 'categorical_crossentropy', 'mae', 'categorical_crossentropy', 'mae'],
#                               loss_weights=[1,1,1,1,1])
#     started_time_date = time.strftime("%Y%m%d_%H%M%S")
#     folder_name = "%s_%s_%s" % (model.name, data.name, started_time_date)
#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name)
#     f = None
#     wr = None
#     started_time = time.time()
#     A = np.zeros((batch_size, model.encoder.vector_size))
#     for i in range(max_iter):
#         sampled_input_images, sampled_target_images, pose_info = data.get_batched_data(batch_size, single_model=False, one_hot=True)
#         input_normalized = model.pixel_normalizer(sampled_input_images)
#         target_normalized = model.pixel_normalizer(sampled_target_images)
#         input_data = [input_normalized, pose_info[1], target_normalized, pose_info[3]]
#         target_data = [input_normalized, pose_info[1], target_normalized, pose_info[3], A]
#
#         loss_info = model.get_model().train_on_batch(input_data, target_data)
#
#         if i % export_image_per == 0:
#             test_few_models_and_export_image(model, data, i, folder_name, test_n=5, single_model=False)
#
#         elapsed_time = time.time() - started_time
#         if i % 100 == 0:
#             if wr is None:
#                 import csv
#                 f = open('%s/log_%s.csv' % (folder_name, started_time_date), 'w', encoding='utf-8')
#                 wr = csv.writer(f)
#                 wr.writerow(["epoch"] + model.get_model().metrics_names + ["elapsed_time"])
#             wr.writerow([i] + loss_info + [time.strftime("%H:%M:%S", time.gmtime(elapsed_time))])
#             # f.write("%f\t%s\n" % (d_loss[1], '\t'.join([str(x) for x in a_loss])))
#             f.flush()
#
#     if save_model:
#         model.save_model('%s/%s' % (folder_name, data.name), started_time_date)

from olds.model.model_Tatarchenko15_attention import ModelTatarchenko15Attention
from olds.model.model_Tatarchenko15_attention2 import ModelTatarchenko15Attention2
from olds.model.model_Zhou16_attention import ModelZhou16Attention
from olds.model.model_Zhou16_attention2 import ModelZhou16Attention2

from olds.model.model_m2n_pg import ModelM2NPixelGeneration
from olds.model.model_m2n_flow import ModelM2NFlow


def get_model_from_info(x, IMSIZE=256):
    model_info, gpu_id, use_data, additional_name = x
    model_type, model_i, k = model_info

    target = ModelZhou16Attention
    if model_type == 'z':
        target = ModelZhou16Attention
    elif model_type == 'z2':
        target = ModelZhou16Attention2
    elif model_type == 't':
        target = ModelTatarchenko15Attention
    elif model_type == 't2':
        target = ModelTatarchenko15Attention2
    elif model_type == 'mp':
        target = ModelM2NPixelGeneration
    elif model_type == 'mf':
        target = ModelM2NFlow

    print("Additional Name is", additional_name)
    additional_name_str = str(additional_name)
    model = None

    if model_i == 1:
        model = target(image_size=IMSIZE, attention_strategy='no', additional_name=additional_name_str)
    elif model_i == 2:
        model = target(image_size=IMSIZE, attention_strategy='u_net', additional_name=additional_name_str)
    elif model_i == 3:
        model = target(image_size=IMSIZE, attention_strategy='cr_attn', mix_concat='concat', k=k, additional_name=additional_name_str)
    elif model_i == 4:
        model = target(image_size=IMSIZE, attention_strategy='cr_attn', mix_concat='mix', k=k, additional_name=additional_name_str)
    elif model_i == 5:
        model = target(image_size=IMSIZE, attention_strategy='mixed', additional_name=additional_name_str, mix_concat='concat',k=2,
                       attention_strategy_details={
                           8:'cr',
                           16:'cr',
                           32:'cr',
                           64: 'cr'
                       })
    elif model_i == 6:
        model = target(image_size=IMSIZE, attention_strategy='double', additional_name=additional_name_str)
    elif model_i == 7:
        model = target(image_size=IMSIZE, attention_strategy='s_attn', mix_concat='mix', k=k, additional_name=additional_name_str)
    elif model_i == 8:
        model = target(image_size=IMSIZE, attention_strategy='hu_attn', additional_name=additional_name_str)
    elif model_i == 9:
        model = target(image_size=IMSIZE, attention_strategy='h_attn', additional_name=additional_name_str)
    elif model_i == 10:
        model = target(image_size=IMSIZE, attention_strategy='u_attn', additional_name=additional_name_str)

    elif type(model_i) == tuple:
        if len(model_i) == 3:
            layer, g, u = model_i
            details = {}
            if '1' in layer:
                details[8] = g
            if '2' in layer:
                details[16] = g
            if '3' in layer:
                details[32] = g
            if '4' in layer:
                details[64] = g
            if '5' in layer:
                details[128] = g
            model = target(image_size=IMSIZE, attention_strategy='mixed', additional_name=additional_name_str,
                           attention_strategy_details=details, u_value=u)
        else:
            layer, g = model_i
            details = {}
            if '1' in layer:
                details[8] = g
            if '2' in layer:
                details[16] = g
            if '3' in layer:
                details[32] = g
            if '4' in layer:
                details[64] = g
            if '5' in layer:
                details[128] = g
            model = target(image_size=IMSIZE, attention_strategy='mixed', additional_name=additional_name_str,
                           attention_strategy_details=details)

    elif model_i == 19:
        model = target(image_size=IMSIZE, attention_strategy='mixed', additional_name=additional_name_str,
                       attention_strategy_details={
                           8:'cr',
                           16:'h',
                           32:'h',
                           64:'h',
                           128:'h'
                       })
    elif model_i == 20:
        model = target(image_size=IMSIZE, attention_strategy='mixed', additional_name=additional_name_str,
                       attention_strategy_details={
                           8:'cr',
                           16:'cr',
                           32:'h',
                           64:'h',
                           128:'h'
                       })
    elif model_i == 21:
        model = target(image_size=IMSIZE, attention_strategy='mixed', additional_name=additional_name_str,
                       attention_strategy_details={
                           8:'cr',
                           16:'cr',
                           32:'cr',
                           64:'h',
                           128:'h'
                       })

    elif model_i == 22:
        model = target(image_size=IMSIZE, attention_strategy='mixed', additional_name=additional_name_str,
                       attention_strategy_details={
                           8:'u_attn',
                           16:'u_attn',
                           32:'u_attn',
                           64:'h',
                           128:'h'
                       })

    elif model_i == 23:
        model = target(image_size=IMSIZE, attention_strategy='mixed', additional_name=additional_name_str,
                       attention_strategy_details={
                           16:'u_net',
                           32:'u_net',
                           64:'h',
                           128:'h'
                       })
    elif model_i == 24:
        model = target(image_size=IMSIZE, attention_strategy='mixed', additional_name=additional_name_str,
                       attention_strategy_details={
                           16:'u_net',
                           32:'u_net',
                           128:'h'
                       })
    return model


def find_load_model_in_folder(model, parent_folder, use_data):
    import glob
    print(model.name)
    files = glob.glob("%s/%s_%s*/*.h5" % (parent_folder, model.name, use_data))
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





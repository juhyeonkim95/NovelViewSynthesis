from keras.layers import Dense, Input, ReLU, Lambda
from keras.layers import Conv2D, Flatten, Concatenate, Activation
from keras.layers import Reshape, Conv2DTranspose, BatchNormalization
from keras.layers import concatenate, Add, Multiply
from olds.model.model_interface import *
from olds.test_utils import *
from olds.model.attention_layers import *
from olds.ops import BilinearSamplingLayer


class ModelZhou16Attention(ModelInterface):
    def __init__(self, image_size=256, attention_strategy='cr_attn', attention_strategy_details=None, mix_concat = None, k=8, additional_name=None):
        super().__init__("zhou16attention", image_size)
        self.pixel_normalizer = lambda x: (x - 0.5) * 2
        self.pixel_normalizer_reverse = lambda x: x / 2 + 0.5
        self.prediction_model = None
        self.attention_strategy = attention_strategy
        self.attention_strategy_details = attention_strategy_details
        self.mix_concat = mix_concat
        self.k = k

        self.name = "%s_%s" % (self.name, self.attention_strategy)
        if self.attention_strategy == 'cr_attn' or self.attention_strategy == 's_attn':
            self.name = "%s_%s_%d" % (self.name, self.mix_concat, self.k)

        if attention_strategy_details is not None:
            for k in sorted(attention_strategy_details.keys()):
                self.name = "%s_%d_%s" % (self.name, k, attention_strategy_details[k])

        if additional_name is not None:
            self.name = "%s_%s" % (self.name, additional_name)

    def build_model(self):
        image_size = self.image_size
        activation = 'relu'
        current_image_size = image_size
        image_input = Input(shape=(current_image_size, current_image_size, 3), name='image_input')
        image_input_normalized = Lambda(self.pixel_normalizer)(image_input)

        i = 0
        x = image_input_normalized

        input_hidden_layers = {}
        while current_image_size > 4:
            x = Conv2D(16 * (2 ** i), kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
            x = Activation(activation=activation)(x)
            i = i+1
            current_image_size = int(current_image_size / 2)
            input_hidden_layers[current_image_size] = x

        x = Flatten()(x)
        hidden_layer_size = int(4096 / 256 * image_size)
        x = Dense(hidden_layer_size, activation=activation)(x)

        viewpoint_input = Input(shape=(18, ), name='viewpoint_input')

        v = Dense(128, activation=activation)(viewpoint_input)
        v = Dense(256, activation=activation)(v)

        concatenated = concatenate([x, v])
        concatenated = Dense(hidden_layer_size, activation=activation)(concatenated)
        concatenated = Dense(hidden_layer_size, activation=activation)(concatenated)

        d = Reshape((4, 4, int(hidden_layer_size / 16)))(concatenated)
        custom_attn_layers = {}

        while current_image_size < image_size / 2:
            current_image_size = current_image_size * 2
            # d = CrossAttentionLayer(input_h=custom_layers[current_image_size])(d)
            current_attention_strategy = self.attention_strategy
            if self.attention_strategy_details is not None:
                current_attention_strategy = self.attention_strategy_details.get(current_image_size,
                                                                                 current_attention_strategy)
            if current_attention_strategy == 'h_attn' or current_attention_strategy == 'h' or current_attention_strategy == 'hu_attn'\
                or current_attention_strategy == 'hu':
                pred_flow = Conv2DTranspose(2, kernel_size=(3, 3), strides=(2, 2), padding='same')(d)

            d = Conv2DTranspose(4 * (2**i), kernel_size=(3, 3), strides=(2, 2), padding='same')(d)
            d = Activation(activation=activation)(d)
            i = i-1

            if current_attention_strategy == 'u_net':
                d = Concatenate()([input_hidden_layers[current_image_size], d])
            elif current_attention_strategy == 'cr_attn'or current_attention_strategy == 'cr':
                c = AttentionLayer(input_h=input_hidden_layers[current_image_size], mix_concat=self.mix_concat, k=self.k)
                custom_attn_layers[current_image_size] = c
                d = c(d)
            elif current_attention_strategy == 's_attn':
                c = AttentionLayer(input_h=d, mix_concat=self.mix_concat, k=self.k)
                custom_attn_layers[current_image_size] = c
                d = c(d)
            elif current_attention_strategy == 'h_attn' or current_attention_strategy == 'h':
                pred_feature = BilinearSamplingLayer(current_image_size)([input_hidden_layers[current_image_size], pred_flow])
                d = Concatenate()([pred_feature, d])
            elif current_attention_strategy== 'u_attn':
                channels = K.int_shape(d)[3] // 2
                print("Channels:", channels)

                g = Conv2D(channels, kernel_size=1, strides=1, padding='same')(d)
                g = BatchNormalization()(g)
                x_original = input_hidden_layers[current_image_size]
                x = Conv2D(channels, kernel_size=1, strides=1, padding='same')(x_original)
                x = BatchNormalization()(x)

                psi = ReLU()(Add()([g, x]))
                psi = Conv2D(1, kernel_size=1, strides=1, padding='same')(psi)
                psi = BatchNormalization()(psi)
                psi = Activation('sigmoid')(psi)

                y = Multiply()([x_original, psi])

                d = Concatenate()([y, d])
            elif current_attention_strategy == 'hu_attn':
                pred_feature = BilinearSamplingLayer(current_image_size)([input_hidden_layers[current_image_size], pred_flow])
                channels = K.int_shape(d)[3] // 2
                print("Channels:", channels)
                g = Conv2D(channels, kernel_size=1, strides=1, padding='same')(d)
                g = BatchNormalization()(g)
                x_original = pred_feature
                x = Conv2D(channels, kernel_size=1, strides=1, padding='same')(x_original)
                x = BatchNormalization()(x)

                psi = ReLU()(Add()([g, x]))
                psi = Conv2D(1, kernel_size=1, strides=1, padding='same')(psi)
                psi = BatchNormalization()(psi)
                psi = Activation('sigmoid')(psi)

                y = Multiply()([x_original, psi])

                d = Concatenate()([y, d])

        self.custom_attn_layers = custom_attn_layers

        pred_flow = Conv2DTranspose(2, kernel_size=(3, 3), strides=(2, 2), padding='same')(d)
        pred_image = BilinearSamplingLayer(self.image_size)([image_input, pred_flow])
        # output = Lambda(self.pixel_normalizer_reverse, name='main_output')(pred_image)

        model = Model(inputs=[image_input, viewpoint_input], outputs=[pred_image])
        model.summary()

        self.model = model

    def set_prediction_model(self):
        # source_image = Input(shape=(self.image_size, self.image_size, 3), name='source_image')
        # target_pose = Input(shape=(5, ), name='target_pose')
        # output = self.get_model()([source_image, target_pose])
        # prediction_model = Model(inputs=[source_image, target_pose], outputs=[output])
        # prediction_model.compile(optimizer="adam", loss="mae", metrics=["mae", ssim_custom])
        self.prediction_model = self.get_model()#prediction_model
        #self.prediction_model.compile(optimizer="adam", loss="mae", metrics=["mae", ssim_custom])

    def get_predicted_image(self, sampled_input_data):
        #if self.prediction_model == None:
        #    self.set_prediction_model()
        source_images, pose_info = sampled_input_data
        A = self.model.predict([source_images, pose_info])
        A = np.clip(A, 0, 1)
        return A

    def process_pose_info(self, data: DataContainer, pose_info):
        source_azimuth = pose_info[1]
        target_azimuth = pose_info[3]
        transformation_azimuth = np.remainder(target_azimuth - source_azimuth + data.n_azimuth, data.n_azimuth)
        transformation_azimuth_onehot = np.eye(data.n_azimuth, dtype=np.float32)[transformation_azimuth]
        return transformation_azimuth_onehot

    # def train(self, data: DataContainer, test_data: DataContainer = None, **kwargs):
    #     self.build_model()
    #     target_loss = kwargs.get('loss', 'mse')
    #     if test_data is not None:
    #         self.model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9), loss=target_loss, metrics=["mae", ssim_custom])
    #         self.prediction_model = self.model
    #     else:
    #         self.model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9), loss=target_loss)
    #
    #     max_iter = kwargs.get("max_iter", 1000000)
    #     batch_size = kwargs.get("batch_size", 32)
    #     single_model = kwargs.get("single_model", False)
    #     export_image_per = kwargs.get("export_image_per", max_iter // 100)
    #     save_model = kwargs.get("save_model", True)
    #     save_model_per = kwargs.get("save_model_per", -1)
    #
    #     started_time_date = time.strftime("%Y%m%d_%H%M%S")
    #     folder_name = "%s_%s_%s" % (self.name, data.name, started_time_date)
    #     additional_folder_name = kwargs.get("parent_folder", None)
    #     if additional_folder_name is not None:
    #         folder_name = additional_folder_name + "/" + folder_name
    #
    #     started_time = time.time()
    #     f = None
    #     wr = None
    #     f_test = None
    #
    #     for i in range(max_iter):
    #         source_images, target_images, pose_info = data.get_batched_data(batch_size, single_model=single_model)
    #         target_view = self.process_pose_info(data, pose_info)
    #         loss_info = self.model.train_on_batch([source_images, target_view], target_images)
    #
    #         if i % export_image_per == 0:
    #             if not os.path.exists(folder_name):
    #                 os.makedirs(folder_name)
    #             test_few_models_and_export_image(self, data, i, folder_name, test_n=5, single_model=False)
    #
    #         elapsed_time = time.time() - started_time
    #         if i % 100 == 0:
    #             if wr is None:
    #                 import csv
    #                 f = open('%s/log_%s.csv' % (folder_name, started_time_date), 'w', encoding='utf-8')
    #                 wr = csv.writer(f)
    #                 wr.writerow(["epoch"] + self.get_model().metrics_names + ["elapsed_time"])
    #
    #             wr.writerow([i] + (loss_info if type(loss_info) is list else [loss_info]) + [time.strftime("%H:%M:%S", time.gmtime(elapsed_time))])
    #             f.flush()
    #
    #         if save_model_per != -1 and save_model and i % save_model_per == 0:
    #             if test_data is not None:
    #                 mae, ssim = test_for_all_models_thorough_per_model(test_data, self)
    #                 if f_test is None:
    #                     f_test = open('%s/test_log_%s.txt' % (folder_name, started_time_date), 'w')
    #                     f_test.write('epoch\tmae\tssim\n')
    #                 f_test.write('%d\t%.4f\t%.4f\n' % (i, mae, ssim))
    #                 f_test.flush()
    #
    #             self.save_model('%s/%s' % (folder_name, data.name), '%s_%d' % (started_time_date, i))
    #
    #     if save_model:
    #         self.save_model('%s/%s' % (folder_name, data.name), started_time_date)

    # def train(self, data: DataContainer, **kwargs):
    #     self.build_model()
    #     self.model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9), loss='mse')
    #     max_iter = kwargs.get("max_iter", 1000000)
    #     batch_size = kwargs.get("batch_size", 32)
    #     single_model = kwargs.get("single_model", False)
    #     export_image_per = kwargs.get("export_image_per", max_iter // 100)
    #     save_model = kwargs.get("save_model", True)
    #     save_model_per = kwargs.get("save_model_per", -1)
    #
    #     started_time_date = time.strftime("%Y%m%d_%H%M%S")
    #     folder_name = "%s_%s_%s" % (self.name, data.name, started_time_date)
    #     additional_folder_name = kwargs.get("parent_folder", None)
    #     if additional_folder_name is not None:
    #         folder_name = additional_folder_name + "/" + folder_name
    #
    #     started_time = time.time()
    #     f = None
    #     wr = None
    #
    #     for i in range(max_iter):
    #         source_images, target_images, pose_info = data.get_batched_data(batch_size, single_model=single_model)
    #         target_view = self.process_pose_info(data, pose_info)
    #         loss_info = self.model.train_on_batch([source_images, target_view], target_images)
    #
    #         if i % export_image_per == 0:
    #             if not os.path.exists(folder_name):
    #                 os.makedirs(folder_name)
    #             test_few_models_and_export_image(self, data, i, folder_name, test_n=5, single_model=False)
    #
    #         elapsed_time = time.time() - started_time
    #         if i % 100 == 0:
    #             if wr is None:
    #                 import csv
    #                 f = open('%s/log_%s.csv' % (folder_name, started_time_date), 'w', encoding='utf-8')
    #                 wr = csv.writer(f)
    #                 wr.writerow(["epoch"] + self.get_model().metrics_names + ["elapsed_time"])
    #
    #             wr.writerow([i] + [loss_info] + [time.strftime("%H:%M:%S", time.gmtime(elapsed_time))])
    #             f.flush()
    #         if save_model_per != -1 and save_model and i > 0 and i % save_model_per == 0:
    #             self.save_model('%s/%s' % (folder_name, data.name), '%s_%d' % (started_time_date, i))
    #
    #     if save_model:
    #         self.save_model('%s/%s' % (folder_name, data.name), started_time_date)
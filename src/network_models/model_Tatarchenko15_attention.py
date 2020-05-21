from keras.layers import Dense, Input, LeakyReLU, ReLU, Lambda
from keras.layers import Conv2D, Flatten, Concatenate, Activation
from keras.layers import Reshape, Conv2DTranspose, BatchNormalization
from keras.layers import concatenate, Add, Multiply
from olds.model.model_interface import *
from olds.utils import *
from olds.test_utils import *
from olds.model.attention_layers import *
from olds.ops import BilinearSamplingLayer


class ModelTatarchenko15Attention(ModelInterface):
    def __init__(self, image_size=256, attention_strategy='cr_attn', attention_strategy_details=None, mix_concat='concat', k=8, additional_name=None, as_flow=False):
        super().__init__("tatarchenko15attention", image_size)
        self.pixel_normalizer = lambda x: (x - 0.5) * 1.5
        self.pixel_normalizer_reverse = lambda x: x / 1.5 + 0.5
        self.prediction_model = None
        self.attention_strategy = attention_strategy
        self.attention_strategy_details = attention_strategy_details
        self.mix_concat = mix_concat
        self.k = k
        self.as_flow = as_flow

        self.name = "%s_%s" % (self.name, self.attention_strategy)
        if self.attention_strategy == 'cr_attn' or self.attention_strategy == 's_attn':
            self.name = "%s_%s_%d" % (self.name, self.mix_concat, self.k)

        if attention_strategy_details is not None:
            for k in sorted(attention_strategy_details.keys()):
                self.name = "%s_%d_%s" % (self.name, k, attention_strategy_details[k])

        if self.as_flow:
            self.name = self.name + "_as_flow"

        if additional_name is not None:
            self.name = "%s_%s" % (self.name, additional_name)

    def build_model(self):
        image_size = self.image_size
        activation = LeakyReLU(0.2)
        current_image_size = image_size
        image_input = Input(shape=(self.image_size, self.image_size, 3), name='image_input')

        image_input_normalized = Lambda(self.pixel_normalizer)(image_input)

        i = 0
        x = image_input_normalized

        while current_image_size > 4:
            k = 5 if current_image_size > 32 else 3
            x = Conv2D(16 * (2 ** i), kernel_size=(k, k), strides=(2, 2), padding='same')(x)
            x = LeakyReLU(0.2)(x)
            x = Conv2D(16 * (2 ** i), kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
            x = LeakyReLU(0.2)(x)
            i = i+1
            current_image_size = int(current_image_size / 2)
            self.encoder_original_features[current_image_size] = x

        x = Flatten()(x)
        hidden_layer_size = int(4096 / 256 * image_size)
        x = Dense(hidden_layer_size, activation=activation)(x)

        viewpoint_input = Input(shape=(5, ), name='viewpoint_input')

        v = Dense(64, activation=activation)(viewpoint_input)
        v = Dense(64, activation=activation)(v)
        v = Dense(64, activation=activation)(v)

        concatenated = concatenate([x, v])
        concatenated = Dense(hidden_layer_size, activation=activation)(concatenated)
        concatenated = Dense(hidden_layer_size, activation=activation)(concatenated)
        concatenated = Dense(hidden_layer_size, activation=activation)(concatenated)

        d = Reshape((4, 4, int(hidden_layer_size / 16)))(concatenated)
        custom_attn_layers = {}


        while current_image_size < image_size / 2:
            k = 5 if current_image_size > 32 else 3
            current_image_size = current_image_size * 2

            current_attention_strategy = self.attention_strategy
            if self.attention_strategy_details is not None:
                current_attention_strategy = self.attention_strategy_details.get(current_image_size,
                                                                                 current_attention_strategy)

            if current_attention_strategy == 'h_attn' or current_attention_strategy == 'h' or \
                    current_attention_strategy=='hu_attn' or current_attention_strategy=='hu':
                pred_flow = Conv2DTranspose(2, kernel_size=(k, k), strides=(2, 2), padding='same')(d)
            d = Conv2DTranspose(4 * (2**i), kernel_size=(k, k), strides=(2, 2), padding='same')(d)
            d = LeakyReLU(0.2)(d)

            # d = Conv2D(4 * (2**i), kernel_size=(k, k), strides=(1, 1), padding='same')(d)
            # d = LeakyReLU(0.2)(d)
            # i = i-1

            if current_attention_strategy == 'double':
                d = Conv2D(8 * (2 ** i), kernel_size=(k, k), strides=(1, 1), padding='same')(d)
            else:
                d = Conv2D(4 * (2 ** i), kernel_size=(k, k), strides=(1, 1), padding='same')(d)
            d = LeakyReLU(0.2)(d)
            i = i - 1

            if current_attention_strategy == 'u_net':
                d = Concatenate()([self.encoder_original_features[current_image_size], d])
            elif current_attention_strategy == 'cr_attn'or current_attention_strategy == 'cr':
                c = AttentionLayer(input_h=self.encoder_original_features[current_image_size], mix_concat=self.mix_concat, k=self.k)
                custom_attn_layers[current_image_size] = c
                d = c(d)
            elif current_attention_strategy == 's_attn':
                c = AttentionLayer(input_h=d, mix_concat=self.mix_concat, k=self.k)
                custom_attn_layers[current_image_size] = c
                d = c(d)
            elif current_attention_strategy == 'h_attn'or current_attention_strategy == 'h':
                pred_feature = BilinearSamplingLayer(current_image_size)([self.encoder_original_features[current_image_size], pred_flow])
                self.decoder_original_features[current_image_size] = d
                self.decoder_rearranged_features[current_image_size] = pred_feature
                d = Concatenate()([pred_feature, d])
            elif current_attention_strategy == 'hu_attn':
                pred_feature = BilinearSamplingLayer(current_image_size)([self.encoder_original_features[current_image_size], pred_flow])
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

            elif current_attention_strategy == 'u_attn':
                channels = K.int_shape(d)[3] // 2
                print("Channels:", channels)

                g = Conv2D(channels, kernel_size=1, strides=1, padding='same')(d)
                g = BatchNormalization()(g)
                x_original = self.encoder_original_features[current_image_size]
                x = Conv2D(channels, kernel_size=1, strides=1, padding='same')(x_original)
                x = BatchNormalization()(x)

                psi = ReLU()(Add()([g, x]))
                psi = Conv2D(1, kernel_size=1, strides=1, padding='same')(psi)
                psi = BatchNormalization()(psi)
                psi = Activation('sigmoid')(psi)

                y = Multiply()([x_original, psi])

                self.decoder_original_features[current_image_size] = d
                self.decoder_rearranged_features[current_image_size] = y

                d = Concatenate()([y, d])

                # c = AttentionLayer(input_h=custom_layers[current_image_size], mix_concat=self.mix_concat, k=self.k)
                # custom_attn_layers[current_image_size] = c
                # d = c(d)
        self.custom_attn_layers = custom_attn_layers
        if self.as_flow:
            pred_flow = Conv2DTranspose(2, kernel_size=(3, 3), strides=(2, 2), padding='same')(d)
            pred_image = BilinearSamplingLayer(self.image_size)([image_input, pred_flow])
            model = Model(inputs=[image_input, viewpoint_input], outputs=[pred_image])
        else:
            d = Conv2DTranspose(3, kernel_size=(5, 5), strides=(2, 2), activation='tanh', padding='same')(d)
            output = Lambda(self.pixel_normalizer_reverse, name='main_output')(d)
            model = Model(inputs=[image_input, viewpoint_input], outputs=[output])
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
        return self.model.predict([source_images, pose_info])



    def process_pose_info(self, data: DataContainer, pose_info):
        batch_size = len(pose_info[0])
        target_azimuth = pose_info[3]
        target_elevation = pose_info[2]
        target_azimuth_cos_sin = index_to_sin_cos(target_azimuth, data.n_azimuth)
        target_elevation_cos_sin = index_to_sin_cos(target_elevation, data.n_elevation, loop=False,
                                                    min_theta=data.min_elevation,
                                                    max_theta=data.max_elevation)

        target_view = np.stack((np.ones(batch_size, ), target_azimuth_cos_sin[0], target_azimuth_cos_sin[1],
                                      target_elevation_cos_sin[0], target_elevation_cos_sin[1]), axis=-1)

        return target_view


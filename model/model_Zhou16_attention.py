from keras.layers import Dense, Input, ReLU, Lambda
from keras.layers import Conv2D, Flatten, Concatenate, Activation
from keras.layers import Reshape, Conv2DTranspose, BatchNormalization
from keras.layers import concatenate, Add, Multiply
from model.model_interface import *
from test_utils import *
from model.attention_layers import *
from ops import BilinearSamplingLayer


class ModelZhou16Attention(ModelInterface):
    def __init__(self, image_size=256, attention_strategy='cr_attn', attention_strategy_details=None,
                 mix_concat = 'concat', k=2, additional_name=None, pose_input_size=None, **kwargs):
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
            if type(list(attention_strategy_details.keys())[0]) == str:
                attention_strategy_details_new = {}
                for k, v in attention_strategy_details.items():
                    attention_strategy_details_new[int(k)] = v
                self.attention_strategy_details = attention_strategy_details_new

            for k in sorted(self.attention_strategy_details.keys()):
                self.name = "%s_%d_%s" % (self.name, k, self.attention_strategy_details[k])

        self.u_value = kwargs.get('u_value', None)

        print("u_value", self.u_value)
        if self.u_value is not None:
            self.name = "%s_%.2f" % (self.name, self.u_value)

        if additional_name is not None:
            self.name = "%s_%s" % (self.name, additional_name)

        self.pose_input_size = pose_input_size if pose_input_size is not None else 18

    def build_model(self):
        image_size = self.image_size
        activation = 'relu'
        current_image_size = image_size
        image_input = Input(shape=(current_image_size, current_image_size, 3), name='image_input')
        image_input_normalized = Lambda(self.pixel_normalizer)(image_input)

        i = 0
        x = image_input_normalized

        while current_image_size > 4:
            x = Conv2D(16 * (2 ** i), kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
            x = Activation(activation=activation)(x)
            i = i+1
            current_image_size = int(current_image_size / 2)
            self.encoder_original_features[current_image_size] = x

        x = Flatten()(x)
        hidden_layer_size = int(4096 / 256 * image_size)
        x = Dense(hidden_layer_size, activation=activation)(x)
        x = Dense(hidden_layer_size, activation=activation)(x)

        viewpoint_input = Input(shape=(self.pose_input_size, ), name='viewpoint_input')

        v = Dense(128, activation=activation)(viewpoint_input)
        v = Dense(256, activation=activation)(v)

        concatenated = concatenate([x, v])
        concatenated = Dense(hidden_layer_size, activation=activation)(concatenated)
        concatenated = Dense(hidden_layer_size, activation=activation)(concatenated)

        d = Reshape((8, 8, int(hidden_layer_size / 64)))(concatenated)
        custom_attn_layers = {}
        self.output_hidden_layers = {}

        channel_size = 256
        current_image_size = 8
        while current_image_size < image_size:
            current_image_size = current_image_size * 2
            # d = CrossAttentionLayer(input_h=custom_layers[current_image_size])(d)
            current_attention_strategy = self.attention_strategy
            if self.attention_strategy_details is not None:
                current_attention_strategy = self.attention_strategy_details.get(current_image_size,
                                                                                 current_attention_strategy)
            if current_attention_strategy == 'h_attn' or current_attention_strategy == 'h' or current_attention_strategy == 'hu_attn'\
                or current_attention_strategy == 'hu':
                if current_image_size < self.image_size:
                    pred_flow = Conv2DTranspose(2, kernel_size=(3, 3), strides=(2, 2), padding='same')(d)

            # d = Conv2DTranspose(channel_size, kernel_size=(3, 3), strides=(2, 2), padding='same')(d)
            # d = Activation(activation=activation)(d)
            # i = i-1
            if current_attention_strategy == 'double':
                d = Conv2DTranspose(2 * channel_size, kernel_size=(3, 3), strides=(2, 2), padding='same')(d)
            else:
                d = Conv2DTranspose(channel_size, kernel_size=(3, 3), strides=(2, 2), padding='same')(d)
            d = Activation(activation=activation)(d)
            i = i - 1
            channel_size = channel_size // 2

            if current_image_size < self.image_size:
                if current_attention_strategy == 'u_net':
                    self.decoder_original_features[current_image_size] = d
                    self.decoder_rearranged_features[current_image_size] = self.encoder_original_features[current_image_size]
                    d = Concatenate()([self.encoder_original_features[current_image_size], d])
                elif current_attention_strategy == 'cr_attn'or current_attention_strategy == 'cr':
                    c = AttentionLayer(input_h=self.encoder_original_features[current_image_size], mix_concat=self.mix_concat, k=self.k,
                                       u_value=self.u_value)
                    custom_attn_layers[current_image_size] = c
                    d = c(d)
                elif current_attention_strategy == 's_attn':
                    c = AttentionLayer(input_h=d, mix_concat=self.mix_concat, k=self.k)
                    custom_attn_layers[current_image_size] = c
                    d = c(d)
                elif current_attention_strategy == 'h_attn' or current_attention_strategy == 'h':
                    pred_feature = BilinearSamplingLayer(current_image_size)([self.encoder_original_features[current_image_size], pred_flow])
                    self.decoder_original_features[current_image_size] = d
                    self.decoder_rearranged_features[current_image_size] = pred_feature
                    d = Concatenate()([pred_feature, d])
                elif current_attention_strategy=='u_attn':
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
                self.output_hidden_layers[current_image_size] = d

        self.custom_attn_layers = custom_attn_layers
        pred_flow = Conv2DTranspose(2, kernel_size=(3, 3), strides=(1, 1), padding='same')(d)

        self.pred_flow = pred_flow
        pred_image = BilinearSamplingLayer(self.image_size)([image_input, pred_flow])
        #output = Lambda(self.pixel_normalizer_reverse, name='main_output')(pred_image)

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
        return self.model.predict([source_images, pose_info])

    def process_pose_info(self, data: DataLoader, pose_info):
        if data.name == 'chair' or data.name == 'car':
            source_azimuth = pose_info[1]
            target_azimuth = pose_info[3]
            transformation_azimuth = np.remainder(target_azimuth - source_azimuth + data.n_azimuth, data.n_azimuth)
            transformation_azimuth_onehot = np.eye(data.n_azimuth, dtype=np.float32)[transformation_azimuth]
            return transformation_azimuth_onehot
        else:
            target_view = pose_info[1] - pose_info[0]
            return target_view

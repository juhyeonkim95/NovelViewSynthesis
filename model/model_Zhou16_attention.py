from keras.layers import Dense, Input, ReLU, Lambda
from keras.layers import Conv2D, Flatten, Concatenate, Activation
from keras.layers import Reshape, Conv2DTranspose, BatchNormalization
from keras.layers import concatenate, Add, Multiply
from model.model_interface import *
from test_utils import *
from model.attention_layers import *
from model.utils import BilinearSamplingLayer, get_modified_decoder_layer


class ModelZhou16Attention(ModelInterface):
    def __init__(self,
                 image_size=256,
                 attention_strategy='h_attn',
                 attention_strategy_details=None,
                 mix_concat='concat',
                 additional_name=None,
                 pose_input_size=18):
        super().__init__("zhou16attention", image_size)
        # normalizing strategy is from the original paper.
        self.pixel_normalizer = lambda x: (x - 0.5) * 2
        self.pixel_normalizer_reverse = lambda x: x / 2 + 0.5
        self.prediction_model = None
        self.attention_strategy = attention_strategy
        self.attention_strategy_details = attention_strategy_details
        self.mix_concat = mix_concat

        self.name = "%s_%s" % (self.name, self.attention_strategy)

        if attention_strategy_details is not None:
            if type(list(attention_strategy_details.keys())[0]) == str:
                attention_strategy_details_new = {}
                for k, v in attention_strategy_details.items():
                    attention_strategy_details_new[int(k)] = v
                self.attention_strategy_details = attention_strategy_details_new

            for k in sorted(self.attention_strategy_details.keys()):
                self.name = "%s_%d_%s" % (self.name, k, self.attention_strategy_details[k])

        if additional_name is not None:
            self.name = "%s_%s" % (self.name, additional_name)

        self.pose_input_size = pose_input_size

    def build_model(self):
        # Build Keras model. Tried to follow the original paper as much as possible.
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

        channel_size = 256
        current_image_size = 8
        while current_image_size < image_size:
            current_image_size = current_image_size * 2
            # d = CrossAttentionLayer(input_h=custom_layers[current_image_size])(d)

            # attention strategy at this layer.
            current_attention_strategy = self.attention_strategy
            if self.attention_strategy_details is not None:
                current_attention_strategy = self.attention_strategy_details.get(current_image_size,
                                                                                 current_attention_strategy)

            # generate flow map t^l from previous decoder layer x^(l+1)_d
            pred_flow = None
            if current_attention_strategy == 'h_attn' or current_attention_strategy == 'h':
                if current_image_size < self.image_size:
                    pred_flow = Conv2DTranspose(2, kernel_size=(3, 3), strides=(2, 2), padding='same')(d)

            # generate next decoder layer x^(l)_d from previous decoder layer x^(l+1)_d
            d = Conv2DTranspose(channel_size, kernel_size=(3, 3), strides=(2, 2), padding='same')(d)
            d = Activation(activation=activation)(d)
            i = i - 1
            channel_size = channel_size // 2

            if current_image_size < self.image_size:
                x_d0 = d
                x_e = self.encoder_original_features[current_image_size]
                x_e_rearranged, x_d = get_modified_decoder_layer(x_d0, x_e, current_attention_strategy, current_image_size, pred_flow)
                self.decoder_original_features[current_image_size] = x_d0
                self.decoder_rearranged_features[current_image_size] = x_e_rearranged
                d = x_d

        # final flow
        pred_flow = Conv2DTranspose(2, kernel_size=(3, 3), strides=(1, 1), padding='same')(d)
        self.pred_flow = pred_flow

        # fetch pixels from original image
        pred_image = BilinearSamplingLayer(self.image_size)([image_input, pred_flow])
        model = Model(inputs=[image_input, viewpoint_input], outputs=[pred_image])
        model.summary()

        self.model = model

    def process_pose_info(self, data: DataLoader, pose_info):
        # Zhou16 used one hot encoded angle value as an input
        if data.name == 'chair' or data.name == 'car':
            source_azimuth = pose_info[1]
            target_azimuth = pose_info[3]
            transformation_azimuth = np.remainder(target_azimuth - source_azimuth + data.n_azimuth, data.n_azimuth)
            transformation_azimuth_onehot = np.eye(data.n_azimuth, dtype=np.float32)[transformation_azimuth]
            return transformation_azimuth_onehot
        else:
            target_view = pose_info[1] - pose_info[0]
            return target_view

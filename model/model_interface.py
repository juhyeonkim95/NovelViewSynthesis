from keras.models import Model
import os
from test_utils import *
from keras.optimizers import Adam
import time

class ModelInterface:
    def __init__(self, name, image_size):
        self.name = name
        self.model = Model()
        self.pixel_normalizer = lambda x:2 * x - 1
        self.pixel_normalizer_reverse = lambda x:0.5 * x + 0.5
        self.image_size = image_size
        self.decoder_original_features = {}
        self.encoder_original_features = {}
        self.decoder_rearranged_features = {}

    def build_model(self):
        pass

    def get_model(self):
        return self.model

    def save_model(self, data_name, file_name):
        self.model.save_weights("%s_%s_%s.h5" % (data_name, self.name, file_name))
        print("Saved model to disk")

    def load_model(self, file_name):
        if not file_name.endswith(".h5"):
            file_name += ".h5"
        self.model.load_weights(file_name)
        print("Loaded model from disk")

    def get_predicted_image(self, sampled_input_data):
        source_images, pose_info = sampled_input_data
        return self.model.predict([source_images, pose_info])

    def process_pose_info(self, data: DataLoader, pose_info):
        return pose_info

    def train(self, data: DataLoader, test_data: DataLoader=None, **kwargs):
        self.build_model()
        target_loss = kwargs.get('loss', 'mae')
        lr = kwargs.get('lr', 0.0001)
        self.model.compile(optimizer=Adam(lr=lr, beta_1=0.9), loss=target_loss)

        max_iter = kwargs.get("max_iterate", 1000000)
        batch_size = kwargs.get("batch_size", 32)
        single_model = kwargs.get("single_model", False)
        export_image_per = kwargs.get("export_image_per", max_iter // 100)
        save_model = kwargs.get("save_model", True)
        save_model_per = kwargs.get("save_model_per", -1)
        write_log_per = kwargs.get("write_log_per", 100)

        started_time_date = time.strftime("%Y%m%d_%H%M%S")
        folder_name = "%s_%s_%s" % (self.name, data.name, started_time_date)
        additional_folder_name = kwargs.get("parent_folder", None)
        if additional_folder_name is not None:
            if not os.path.exists(additional_folder_name):
                os.makedirs(additional_folder_name)
            folder_name = additional_folder_name + "/" + folder_name

        started_time = time.time()
        f = None
        wr = None
        f_test = None

        for i in range(max_iter):
            source_images, target_images, pose_info = data.get_batched_data(batch_size, single_model=single_model, is_train=True)
            target_view = self.process_pose_info(data, pose_info)
            loss_info = self.model.train_on_batch([source_images, target_view], target_images)

            if i % export_image_per == 0:
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                test_few_models_and_export_image(self, data, str(i), folder_name, test_n=5, single_model=False)

            elapsed_time = time.time() - started_time

            # Write log.
            if i % write_log_per == 0:
                if wr is None:
                    import csv
                    f = open('%s/log_%s.csv' % (folder_name, started_time_date), 'w', encoding='utf-8')
                    wr = csv.writer(f)
                    wr.writerow(["epoch"] + self.get_model().metrics_names + ["elapsed_time"])

                wr.writerow([i] + (loss_info if type(loss_info) is list else [loss_info]) +
                            [time.strftime("%H:%M:%S", time.gmtime(elapsed_time))])
                f.flush()

            if save_model_per != -1 and save_model and i % save_model_per == 0:
                if test_data is not None:
                    mae, ssim = test_for_all_models_thorough_per_model(test_data, self)
                    if f_test is None:
                        f_test = open('%s/test_log_%s.txt' % (folder_name, started_time_date), 'w')
                        f_test.write('epoch\tmae\tssim\n')
                    f_test.write('%d\t%.4f\t%.4f\n' % (i, mae, ssim))
                    f_test.flush()
                self.save_model('%s/%s' % (folder_name, data.name), '%s_%d' % (started_time_date, i))

        if save_model:
            self.save_model('%s/%s' % (folder_name, data.name), started_time_date)

    def evaluate(self, source_images, target_images, pose_info):
        if self.prediction_model is None:
            self.prediction_model = self.get_model()  # prediction_model
            self.prediction_model.compile(optimizer="adam", loss="mae", metrics=[mae_custom, ssim_custom])
        return self.prediction_model.evaluate([source_images, pose_info], target_images, verbose=False)

from machine_learning.models import RegressionModels as Regs
from machine_learning.dataset_generators import DatasetSingleFrame as Dataset, CracksPlanar as Features, \
    Displacements as Labels
from machine_learning.callbacks import TrainingProgress as TP
from callback_legacy import LossHistory
from keras.optimizers import RMSprop, Adam, Nadam
from keras.callbacks import ModelCheckpoint
from sklearn import preprocessing as pre
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense

checkpoint_filepath = "ResNet50.hdf5"

cp = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

epochs = 200

dataset = Dataset()

features = Features(dataset, one_hot=True)

print(features.feature_shape)

# Labels
channels_labels = "160"
levels_labels = '12'
result_type = 'all'
result_time = 48
result_column = 1

no_instances = 'all'

labels = Labels(dataset, channels=channels_labels, result_time=result_time, result_type=result_type,
                levels=levels_labels)

min_max_scaler = pre.MinMaxScaler(feature_range=(0, 1))

# features.transform(min_max_scaler)
labels.transform(min_max_scaler)

sh = TP(features, labels, plot_back=3)

extra_layers = (Dropout(0.3), )
model = Regs.vgg_model(features.feature_shape, labels.label_shape, final_bias=True, extra_layers=extra_layers)

# extra_layers = (Dropout(0.3), BatchNormalization())
# model = Regs.vgg_model(features.feature_shape, labels.label_shape, final_bias=True, extra_layers=extra_layers)

opt = Nadam(0.0001)

model.compile(loss="mse", optimizer=opt)

output = model.fit(features.training_set(), labels.training_set(),
                   validation_data=(features.validation_set(), labels.validation_set()), epochs=epochs,
                   callbacks=[sh], )

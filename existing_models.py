from machine_learning.models import RegressionModels as Regs
from machine_learning.dataset_generators import DatasetSingleFrame as Dataset, CracksPlanar as Features, \
    Displacements as Labels
from machine_learning.callbacks import SimpleHistory as SH
from keras.optimizers import RMSprop, Adam, Nadam
from keras.callbacks import ModelCheckpoint
from sklearn import preprocessing as pre

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

sh = SH(plot_from=5, plot_every_n_epochs=25, plot_prev_n_epochs=3, features=features, labels=labels)

model = Regs.existing(features.feature_shape, labels.label_shape, final_activation='relu',
                      final_bias=False)

opt = Nadam(0.05)

model.compile(loss="mse", optimizer=opt)

output = model.fit(features.training_set(), labels.training_set(),
                   validation_data=(features.validation_set(), labels.validation_set()), epochs=epochs,
                   callbacks=[cp, sh])

loss = output.history['loss']
val_loss = output.history['val_loss']

print("minimum loss:", min(loss))

print("minimum val loss:", min(val_loss))




from machine_learning.dataset_generators import DatasetSingleFrame, Cracks1D, Displacements
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from sklearn import preprocessing as pre
from keras.optimizers import RMSprop, Adam, Nadam
from keras.losses import mean_squared_error as loss

dataset = DatasetSingleFrame()
cracks = Cracks1D(dataset)
displacements = Displacements(dataset, result_type="all", unit="milimeters", result_time=48)

min_max_scaler = pre.MinMaxScaler(feature_range=(0, 1))

cracks.transform(min_max_scaler)

displacements.transform(min_max_scaler)

print(displacements.training_set())

#
# # define the keras model
model = Sequential()
model.add(Dense(32, input_dim=displacements.label_shape, activation='relu'))

model.add(Dense(32, input_dim=displacements.label_shape, activation='sigmoid'))
# # compile the keras model
model.compile(loss=['categorical_crossentropy', 'mse'], optimizer=Nadam(0.005), metrics=['accuracy'])

# model.summary()
# # # # fit the keras model on the dataset
# model.fit(displacements.training_set(), cracks.training_set(),
#           validation_data=(displacements.validation_set(), cracks.validation_set()),
#           epochs=50, verbose=2)



# make class predictions with the model
predictions = model.predict(displacements.training_set())

for prediction in predictions:
    print(prediction)

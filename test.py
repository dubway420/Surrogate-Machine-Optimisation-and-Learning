from parmec_analysis.reactor_case import Parse as ps
from machine_learning.dataset_generators import DatasetSingleFrame, Features, min_max_channels_levels
from parmec_analysis.utils import is_in
import numpy as np

case1 = ps("training_data_sub/batch13_36424_P40/batch13_36424_P40")

(min_channel, max_channel), (min_level, max_level) = min_max_channels_levels(case1, "1-3", "all")

# print(min_channel, max_channel, min_level, max_level)


# for i in range(min_channel + 1, max_channel + 1):
#     print(i)

for i in range(0, 5):
    array = case1.get_channel_crack_array(160, array_size=i, array_type="pos", channel_type="inter")
    print(max(i*2, 1))

# for i in range(1, case1.interstitial_channels + 1):
#     # print(case1.get_channel_crack_array((i+1), array_type="pos", channel_type="inter"))
#     print("Channel:", i, "Distance from centre:", case1.distance_from_centre(i, channel_type="inter"))

class FeaturesChannelsUnique3D(Features):

    def __init__(self, dataset, channels='all', levels='all', array_type='positions only', array_size=2,
                 extra_dimension=False):

        self.feature_mode = "3D_unique"

        self.extra_dimension = extra_dimension

        super().__init__(dataset, channels, levels, array_type)

        # the super class tries to load features from files. If it fails, the features are loaded from the dataset
        if self.values is None:
            self.values = self.generate_array(dataset, channels, levels, array_type, array_size)

        # if self.extra_dimension:
        #     self.feature_shape = self.values.shape[1:]
        # else:
        #     self.feature_shape = self.values.shape[1]

    def generate_array(self, dataset, channels, levels, array_type, array_size):


        # instance_array_shape =
        shape_convo = [len(dataset.cases_list) * self.number_channels, 35]

        if is_in(array_type, "orien"):
            shape_convo.append(4)

        X_3du = np.zeros(shape_convo)

        # for i, instance in enumerate(dataset.core_instances):
        #     for c in range()
        #     # crack_array = instance.crack_array_1d(channels=channels, levels=levels, array_type=array_type)
        # if is_in(array_type, "pos"):
        #     X_1d[i] = crack_array
        # else:
        #     for j in range(1, 5):
        #         X_1d[i, :, j - 1] = np.where(crack_array == float(j), 1, 0)

        # If the user requires the array to be configured for convolutional networks, then the array is reshaped
        # if extra_dimension:
        #     if is_in(array_type, "pos"):
        #         X_shape = [X_1d.shape[0], X_1d.shape[1], 1]
        #         return X_1d.reshape(X_shape)
        #     else:
        #         X_shape = [X_1d.shape[0], X_1d.shape[1], X_1d.shape[2], 1]
        #         return X_1d.reshape(X_shape)
        #
        # # Else just return the normal numpy array
        # else:
        #     return X_1d


no_instances = 'all'
#
dataset = DatasetSingleFrame('training_data_sub/', number_of_cases=no_instances)

features = FeaturesChannelsUnique3D(dataset)

#
# #
# def turn_off_graph_decorations(axis):
#     axis.xaxis.set_ticks_position('none')
#     axis.yaxis.set_ticks_position('none')
#     axis.set_xticklabels([])
#     axis.set_yticklabels([])
#
#
# trial_name = experiment.trial.trial_name
#
# folder_validation(trial_name)
#
# experiment_folder = trial_name + "/" + experiment.name
# folder_validation(experiment_folder)
#
# view = CV(trial_name, 0, experiment)
# view.update_data(1, experiment.model, False, False, True)
# view.plot()

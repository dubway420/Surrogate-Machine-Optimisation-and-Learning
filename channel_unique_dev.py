from machine_learning.dataset_generators import DatasetSingleFrame, Features, Features3D, min_max_channels_levels
from parmec_analysis.utils import is_in
import numpy as np
from parmec_analysis.reactor_case import Parse

case1 = Parse("batch21_7159_P40/batch21_7159_P40")
dataset = DatasetSingleFrame("training_data_sub/")


# print(case1.get_channel_crack_array(1, array_type="pos", channel_type="interstitial", array_size=4).shape)
# print(case1.get_crack_array(array_type="pos").shape)


class FeaturesChannelUniques(Features):

    def __init__(self, dataset, channels='all', levels='all', array_type='positions only', array_size=2,
                 extra_dimension=False):

        self.feature_mode = "3D_unique_" + str(array_size)

        self.extra_dimension = extra_dimension

        if array_size >= dataset.core_instances[0].padding:
            print("You have requested an array size (" + str(array_size) + ") which is greater or equal to the padding "
                                                                           "size around the core array (" +
                  str(dataset.core_instances[0].padding) + "). If this operation fails, consider adjusting the padding "
                                                           "around the core array (see padding arguement in class Parse"
                                                           " in the reactor_case module) then recreate the dataset.")

        self.array_size = array_size

        super().__init__(dataset, channels, levels, array_type)

        # redefine the channels attributes. This is done because the super class calculates them based on fuel channel
        # numbers.
        # ===========
        channels_range, _ = min_max_channels_levels(dataset.core_instances[0], channels, levels,
                                                    channel_type="interstitial")

        self.channels_range = channels_range

        self.number_channels = channels_range[1] - channels_range[0]

        # ===========

        # the super class tries to load features from files. If it fails, the features are loaded from the dataset
        if self.values is None:
            self.values = self.generate_array(dataset, channels, levels, array_type, array_size, extra_dimension)

        self.feature_shape = self.values.shape[1:]

        # self.save()

    def generate_array(self, dataset, channels, levels, array_type, array_size, extra_dimension):

        # if is_in(array_type, "orien"):
        #     shape.append(4)

        X_convo = np.zeros([len(dataset.cases_list) * self.number_channels, self.number_levels, self.array_size * 2,
                            self.array_size * 2])

        for i, instance in enumerate(dataset.core_instances):

            for j in range(self.channels_range[0], self.channels_range[1]):
                channel_instance = (i * self.number_channels) + j
                X_convo[channel_instance] = instance.get_channel_crack_array((j + 1), array_type="pos",
                                                                             channel_type="interstitial",
                                                                             array_size=array_size)

        # If the user requires the array to be configured for convolutional networks, then the array is reshaped
        if extra_dimension:
            X_shape = [X_convo.shape[0], X_convo.shape[1], X_convo.shape[2], X_convo.shape[3], 1]
            return X_convo.reshape(X_shape)

        # Else just return the normal numpy array
        else:
            return X_convo


# features3D = Features3D(dataset)
features = FeaturesChannelUniques(dataset, array_size=2, extra_dimension=False)
print(features.values)


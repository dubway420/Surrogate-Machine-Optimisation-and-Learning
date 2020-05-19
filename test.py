# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import AxesGrid
# from parmec_analysis.visualisation import CoreView as CV
# from experiment_input_files.experiment1 import experiment
# from parmec_analysis.utils import folder_validation
# from parmec_analysis.utils import convert_case_to_channel_result

# from experiment_input_files.trial_common_parameters import parameters
from machine_learning.dataset_generators import FeaturesConcentration1D as Features
from machine_learning.dataset_generators import DatasetSingleFrame

no_instances = 'all'

dataset = DatasetSingleFrame('~/localscratch/', number_of_cases=no_instances)
features = Features(dataset, extra_dimension=False)
#
print(features.feature_shape)

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

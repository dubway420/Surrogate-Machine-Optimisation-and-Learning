from parmec_analysis.dataset_generators import DatasetSingleFrame, Features1D, Features2D, Features3D, Labels
from parmec_analysis.models import RegressionModels as RegMods
from parmec_analysis.experiment import Experiment

# Features
channels_features = 'all'
levels_features = 'all'
array_type = 'Positions Only'

# Labels
channels_labels = 'all'
levels_labels = 'all'
result_type = 'all'
result_time = 48
result_column = 1

no_instances = 'all'

dataset = DatasetSingleFrame('~/localscratch/', number_of_cases=no_instances)

labels = Labels(dataset, result_time=48, result_type='all')

list_of_experiments = [

    # Regularisation
    # Separate model for inner and outer

    # Experiment("MLP", RegMods.multi_layer_perceptron, dataset, Features1D(dataset), labels),                  # 0
    # Experiment("WP", RegMods.wider_model, dataset, Features1D(dataset), labels),                              # 1
    # Experiment("CNN1D_Flat", RegMods.cnn1D, dataset, Features1D(dataset, extra_dimension=True), labels),      # 2
    Experiment("CNN1D_Multi", RegMods.cnn1D, dataset, Features2D(dataset), labels),                           # 3
    Experiment("CNN2D_T1", RegMods.cnn2D_type1, dataset, Features2D(dataset, extra_dimension=True), labels),  # 4
    Experiment("CNN2D_T2", RegMods.cnn2D_type2, dataset, Features2D(dataset, extra_dimension=True), labels),  # 5
    Experiment("CNN3D_T1", RegMods.cnn2D_type1, dataset, Features3D(dataset), labels),                        # 6
    Experiment("CNN3D_T2", RegMods.cnn2D_type2, dataset, Features3D(dataset), labels)                         # 7

]

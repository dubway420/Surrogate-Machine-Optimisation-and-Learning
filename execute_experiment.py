from machine_learning.model_training import run_experiment
import sys

package = "experiment_input_files.experiment" + sys.argv[-1]

experiment = getattr(__import__(package, fromlist=["experiment"]), "experiment")

run_experiment(experiment)

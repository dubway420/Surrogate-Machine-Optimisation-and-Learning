from machine_learning.model_training import run_experiment
import sys

trial = sys.argv[-1]

exp_number = sys.argv[-2]

package = trial + ".experiment_input_files.experiment" + exp_number

experiment = getattr(__import__(package, fromlist=["experiment"]), "experiment")

run_experiment(experiment(trial))

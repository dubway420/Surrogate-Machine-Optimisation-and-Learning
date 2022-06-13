from keras.utils.vis_utils import plot_model
import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
import matplotlib as mpl


def summary(experiment):
    experiment_attr = [experiment.trial, experiment.dataset, experiment.features, experiment.labels]

    model = experiment.model

    output_file_path = experiment.trial.trial_name + "/" + experiment.name + "/" + experiment.name

    f = open('changelog.txt')

    version = f.readline()

    f.close()

    with open(output_file_path + ".info", 'w') as fh:

        fh.write("Summary of experiment: " + experiment.name + "\n\n")

        for attr in experiment_attr:

            summary = attr.summary()

            for line in summary:
                fh.write(line + "\n")

            fh.write("\n")

        # Pass the file handle in as a lambda function to make it callable
        # model.summary(print_fn=lambda x: fh.write(x + '\n'))

        fh.write("Machine Learning Model Parameters:\n")
        fh.write(model.name + "\n")
        fh.write("Batch Size: " + str(experiment.batch_size) + "\n\n")

        for layer in model.layers:
            config = layer.get_config()
            for term in config:
                fh.write(term + ": " + str(config[term]) + "\n")
            fh.write("\n")

        fh.write("\n--------------------\n")

        fh.write("This model was trained using the Machine Learning Surrogate Model Framework by H.Rhys Jones\n")
        fh.write("Copyright 2022\n")
        fh.write("contact: huw.jones@manchester.ac.uk\n")

        fh.write("\n")
        fh.write(version)
        fh.write("\n")
        message = "Tensorflow version: " + tf.__version__
        fh.write(message)
        fh.write("\n")
        message = "Keras version:" + tfk.__version__
        fh.write(message)
        fh.write("\n")
        message = "Numpy version: " + np.__version__
        fh.write(message)
        fh.write("\n")
        message = "Matplotlib version: " + mpl.__version__
        fh.write(message)






    plot_file_name = output_file_path + ".png"
    # plot_model(model, to_file=plot_file_name, show_shapes=True, show_layer_names=True)




from keras.utils.vis_utils import plot_model


def summary(experiment):
    experiment_attr = [experiment.trial, experiment.dataset, experiment.features, experiment.labels]

    model = experiment.model

    output_file_path = experiment.trial.trial_name + "/" + experiment.name + "/" + experiment.name

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
        fh.write(model.name + "\n\n")

        for layer in model.layers:
            config = layer.get_config()
            for term in config:
                fh.write(term + ": " + str(config[term]) + "\n")
            fh.write("\n")

    plot_file_name = output_file_path + ".png"
    plot_model(model, to_file=plot_file_name, show_shapes=True, show_layer_names=True)



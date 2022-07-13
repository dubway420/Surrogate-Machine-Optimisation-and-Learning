from decimal import Decimal


class TrialParameters:

    def __init__(self, trial_name, dataset, labels, epochs, optimiser, loss_function, plot_every_n_epochs, save_model):
        self.trial_name = trial_name
        self.dataset = dataset
        self.labels = labels
        self.epochs = epochs
        self.optimiser = optimiser
        self.loss_function = loss_function
        self.plot_every_n_epochs = plot_every_n_epochs
        self.save_model = save_model

    def summary(self):
        summary_text = [

            "Part of trial: " + self.trial_name,
            "Training for: " + str(self.epochs) + " epochs",
            "Optimiser: " + type(self.optimiser).__name__,
            "Learning rate: " + '%.2E' % Decimal(self.optimiser.get_config()['learning_rate']),
            "Decay: " + '%.2E' % Decimal(self.optimiser.get_config()['decay']),
            "Loss function: " + self.loss_function.__name__

        ]

        return summary_text

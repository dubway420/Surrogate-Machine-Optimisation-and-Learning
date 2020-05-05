class TrialParameters:

    def __init__(self, trial_name, dataset, labels, epochs, optimiser, loss_function, plot_every_n_epochs):

        self.trial_name = trial_name
        self.dataset = dataset
        self.labels = labels
        self.epochs = epochs
        self.optimiser = optimiser
        self.loss_function = loss_function
        self.plot_every_n_epochs = plot_every_n_epochs


class Experiment:

    def __init__(self, trial, name, model, dataset, features, labels):

        self.trial = trial

        self.name = name
        self.model = model
        self.dataset = dataset
        self.features = features
        self.labels = labels

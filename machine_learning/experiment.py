class Experiment:

    def __init__(self, trial, name, model, dataset, features, labels, batch_size=32, callbacks=[]):

        self.trial = trial

        self.name = name
        self.model = model
        self.dataset = dataset
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.callbacks = callbacks

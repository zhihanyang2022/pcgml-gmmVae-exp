class Learner():
    def __init__(self, train_data, model, loss, optim, valid_data=None):
        self.train_data, self.model, self.loss, self.optim, self.valid_data = train_data, model, loss, optim, valid_data
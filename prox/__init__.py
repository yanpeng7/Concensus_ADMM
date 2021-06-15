class Proximal:
    def __init__(self):
        self.lambda_ = 0

    def prox(self, x, **kwargs):
        pass

    def eval(self, x, **kwargs):
        pass


class NoPrior(Proximal):
    def __init__(self):
        super().__init__()

        self.lambda_ = 0

    def prox(self, x, **kwargs):
        return x

    def eval(self, x, **kwargs):
        return 0

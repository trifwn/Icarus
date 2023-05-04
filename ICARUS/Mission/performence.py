class Fitness:
    def __init__(self, func):
        self.func = func

    def getFitness(self, *args, **kwargs):
        return self.func(*args, **kwargs)

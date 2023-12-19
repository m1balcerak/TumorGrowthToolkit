class BaseSolver:
    def __init__(self, params):
        self.params = params

    def solve(self):
        raise NotImplementedError("Solve method must be implemented by the subclass.")


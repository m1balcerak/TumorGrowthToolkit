class BaseSolver:
    def __init__(self, params):
        self.params = params
        # Common initialization for all solvers

    def solve(self):
        raise NotImplementedError("Solve method must be implemented by the subclass.")


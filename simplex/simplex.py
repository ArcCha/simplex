class Simplex(object):
    """Solver for linear problems given in a standard form."""

    def __init__(self, c, A, b):
        """
        Accepts linear problem in following form:

        Maximize c^T * x, subject to Ax <= b, all x_i >= 0.

        Args:
            c (ndarray): 1-D vector of objective function coefficients we want to maximize.
            A (ndarray): 2-D matrix of constraints coefficients.
            b (ndarray): 1-D vector of constraints constants.
        """
        self.c = c
        self.A = A
        self.b = b
        self.tableau = None


        def solve(self):
            self._compute_canonical_tableau()
            while not self._is_solution_optimal():
                pivot = self._select_pivot()
                self._optimize_around(pivot)

        def _compute_canonical_tableau(self):
            pass

        def _is_solution_optimal(self):
            pass

        def _select_pivot(self):
            """
            Selects pivot around which we would like to perform optimization.

            Returns:
                (i, j): i - row, j - column
            """
            pass


        def _optimize_around(self, pivot):
            pass



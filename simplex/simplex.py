import numpy as np
import numpy.ma as ma


class UnsolvableProblem(RuntimeError):
    pass


class UnboundedProblem(RuntimeError):
    pass


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
        if (c.ndim != 1 or
            A.ndim != 2 or
            b.ndim != 1 or
            c.shape[0] != A.shape[1] or
            A.shape[0] != b.shape[0]):
            raise TypeError('improper shapes of input matrices given')
        self.c = c
        self.A = A
        self.b = b
        self.optim_rhs = 0
        self.tableau = None


    def solve(self):
        self._compute_canonical_tableau()
        while not self._is_solution_optimal():
            pivot = self._select_pivot()
            self._optimize_around(pivot)

    def _compute_canonical_tableau(self):
        S = np.eye(len(self.b))
        self.tableau = np.hstack((self.A, S))

        c_zeros = np.zeros(len(self.b))
        self.c = np.hstack((self.c, c_zeros))

    def _is_solution_optimal(self):
        return np.all(self.c <= 0)

    def _select_pivot(self):
        """
        Selects pivot around which we would like to perform optimization.

        Returns:
            (i, j): i - row, j - column
        """
        # col - most positive value in objective row
        col = np.argmax(self.c) 
        masked_column = ma.MaskedArray(self.tableau[:, col], self.tableau[:, col] <= 0)
        if masked_column.count() == 0:
            raise UnboundedProblem()
        ratios = self.b / masked_column
        # we naively choose the first minimum ratio
        min_idx = ma.argmin(ratios)
        return (min_idx, col)


    def _optimize_around(self, pivot):
        row, col = pivot
        self.tableau[row, :] /= self.tableau[row, col]
        self.b[row] /= self.tableau[row, col]

        pivot_val = self.tableau[row, col]

        # Pivot c-row
        val = self.c[col]
        multiplier = val / pivot_val
        self.c -= self.tableau[row] * multiplier
        self.optim_rhs -= self.tableau[row] * multiplier

        for idx in range(self.tableau.shape[0]):
            if idx == row:
                continue
            val = self.tableau[idx, col]
            multiplier = val / pivot_val
            self.tableau[idx] -= self.tableau[row] * multiplier
            self.b[idx] -= self.b[row] * multiplier

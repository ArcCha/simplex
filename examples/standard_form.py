import os  # isort:skip
import sys  # isort:skip
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip

import numpy as np

from simplex import Simplex


if __name__ == '__main__':
    c = np.array([-2, -3, -4])
    A = np.array([[3, 2, 1], [2, 5, 3]])
    b = np.array([10, 15])
    s = Simplex(c, A, b)
    print(s.solve())


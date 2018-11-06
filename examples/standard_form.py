import os  # isort:skip
import sys  # isort:skip
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip

import numpy as np

from simplex import Simplex


if __name__ == '__main__':
    c = np.array([2, -3, -4]).astype("float")
    A = np.array([[3, 2, 1], [2, 5, 3]]).astype("float")
    b = np.array([10, 15]).astype("float")
    s = Simplex(c, A, b)
    print("Before solving", s.A, s.b, s.c)
    print(s.solve())
    print("After solving", s.tableau, s.b, s.c)


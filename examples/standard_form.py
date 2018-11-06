import os  # isort:skip
import sys  # isort:skip
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip

import numpy as np

from simplex import Simplex, UnboundedProblem


if __name__ == '__main__':
    print('Simple, existing solution')
    c = np.array([2, 3, 4])
    A = np.array([[3, 2, 1], [2, 5, 3]])
    b = np.array([10, 15])
    s = Simplex(c, A, b)
    res = s.solve()
    if res == -20:
        print('pass')
    else:
        print('fail ' + str(s.solve()))
    
    print('Unbounded problem')
    c = np.array([5, 4]).astype("float")
    A = np.array([[1, 0], [1, -1]]).astype("float")
    b = np.array([7, 8]).astype("float")
    s = Simplex(c, A, b)
    # print("Before solving", s.A, s.b, s.c)
    thrown = False
    try:
        res = s.solve()
    except UnboundedProblem:
        thrown = True
    if thrown:
        print('pass')
    else:
        print('fail')


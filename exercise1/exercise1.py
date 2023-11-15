"""
Code for Exercise 1

Name: Sam Hanson
Number: s2153833

"""
import numpy as np
import scipy.constants
G = scipy.constants.G


def task1(x, n, b):
    """
    Task 1 is asking to check if b is true and to do a sum of the values in x
    if b is false you sum the values of x**2.
    Parameters
    ----------
    x : [3, 4, 5], string of numbers.
    n : 1, integer value
    b : True/False

    Returns
    -------
    if b is true result = 3 + 4 + 5 = 12
    if b is false result = 3**2 + 4**2 * 5**2 = 50
    """
    result = 0
    for i in x:
        if b:
            result += i**n
        else:
            result += i**(2*n)
    return result


def task2(m, n):
    """
    Task 2 asks to create numpy arrays of size m and n,
    then an array of random numbers of shape (n, m).

    Parameters
    ----------
    m : Size of array "a"
    n : Size of array "b"

    Returns
    -------
    Array of zeroes of size m
    Array of numbers from 0 to n - 1
    Array of random numbers of shape (n, m)

    """
    a = np.zeros(m)
    b = np.arange(1, n + 1)  # changed from (0, n + 1)
    c = np.random.rand(m, n)
    return a, b, c


def task3a(a, b, t):
    """
    Task 3a asks to sum vectors a and b, then multiply by 2t

    Parameters
    ----------
    a : numpy array, a

    b : numpy array, b

    t : float, t

    Returns
    -------
    x : vector sum of 2*t*(a+b)

    """
    x = 2*t*(a + b)
    return x


def task3b(x, y):
    """
    Find the distance between vectors x and y

    Parameters
    ----------
    x : Position vector.
    y : Position vector.
    Returns
    -------
    s : distance between x and y.

    """
    z = x - y
    s = np.linalg.norm(z)
    return s


def task4a(v1, v2):
    """
    v1 cross v2 = -v2 cross v1

    Parameters
    ----------
    v1 : numpy vector
    v2 : numpy vector

    Returns
    -------
    a : numpy vector = v1 cross v2
    b : numpy vector = -1 * v2 cross v1

    """
    a = np.cross(v1, v2)
    b = np.cross(v2, v1)*(-1)
    return a, b


def task4b(v1, v2, v3):
    """
    Check v1 cross (v2 + v3) = (v1 cross v2) + (v1 cross v3)

    Parameters
    ----------
    v1 : numpy vector
    v2 : numpy vector
    v3 : numpy vector

    Returns
    -------
    a2 : numpy vector = v1 cross (v2 + v3)
    b3 : numpy vector = (v1 cross v2) + (v1 cross v3)

    """
    a1 = v2 + v3
    a2 = np.cross(v1, a1)
    b1 = np.cross(v1, v2)
    b2 = np.cross(v1, v3)
    b3 = b1 + b2
    return a2, b3


def task4c(v1, v2, v3):
    """
    Vector triple product of v1, v2 and v3.
    v1 cross (v2 cross v3) = v2*(v1 dot v3) - v3*(v1 dot v2)

    Parameters
    ----------
    v1 : numpy vector
    v2 : numpy vector
    v3 : numpy vector

    Returns
    -------
    a2 : numpy vector = v1 cross (v2 cross v3)
    b3 : numpy vector = v2 * (v1 dot v3) - v3 * (v1 dot v2)

    """
    a1 = np.cross(v2, v3)
    a2 = np.cross(v1, a1)
    b1 = np.dot(v1, v3)
    b2 = np.dot(v1, v2)
    b3 = b1*v2 - b2*v3
    return a2, b3


def task5(x1, x2, M1, M2):
    """
    Calculate the gravitational force of mass 2 on mass 1

    Parameters
    ----------
    x1 : Position vector of mass 1 in m
    x2 : position vector of mass 2 in m
    M1 : Mass 1 in kg
    M2 : Mass 2 in kg

    Returns
    -------
    Gravitational force of x2 on x1
    gravitational potential energy of the system

    """
    # F = GM1M2/r^3 * (x2 - x1) where r = norm(x2 - x1)
    r = np.linalg.norm(x2 - x1)
    f = -1*G*M1*M2*(x2 - x1)/(r**3)
    # V = GM1M2/r
    v = G*M1*M2/r
    return f, v


def task6a(n):
    """
    make a 2D array, M, of shape (n, n) with elements (i, j) = i + 2*j

    Parameters
    ----------
    n : number of rows/columns of M

    Returns
    -------
    M : 2d numpy array

    """
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = i + 2*j
    return M


def task6b(n):
    """
    Using the array from task6a, make a new array of size n
    The elements of the array will be equal to the sum of the rows of M

    Parameters
    ----------
    n : size of array, y.

    Returns
    -------
    y : numpy array, elements = sum of rows of M

    """
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = i + 2*j
    y = np.sum(M, axis=1)
    return y

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 20:21:22 2023

@author: Samha
"""
import math
import matplotlib.pyplot as pyplot


# my_function is the fourier series of a square wave from Task 3b.
def my_function(i, x):
    # sum of 1 to N of sin((2i-1)x)/(2i-1)
    k = 2*i - 1
    return (1/k)*math.sin(k*x)


def main():
    n_loop = int(input("What is your N value?"))
    # exercise0data.txt
    file = str(input("What is the file you want to write to?"))
    out_file = open(file, "w")
    x_data = []
    y_data = []
    for i in range(n_loop):
        x = 2*math.pi*i/n_loop - math.pi
        f = 0
        for j in range(1, i):
            f += my_function(i, x)
        x_data.append(x)
        print(str(f))
        y_data.append(f)
        out_file.write(str(x) + " " + str(f) + "\n")

    out_file.close()

    pyplot.plot(x_data, y_data)
    pyplot.title("Plotting a square wave")
    pyplot.xlabel("x")
    pyplot.show()


main()

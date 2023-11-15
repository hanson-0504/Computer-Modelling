# -*- coding: utf-8 -*-
"""
Plotting code:
    Uses simulate_particle.py to compute frequency inaccuracy as a plot of
    frequency inaccuracy against time-step
    For Euler/Verlet Simulation.
"""
import math
import matplotlib.pyplot as pyplot
import scipy.constants as sc


def main():
    time_units = (
        sc.value("Angstrom star"))*math.sqrt(sc.value(
            "atomic mass constant")/sc.value("electron volt"))

    files = [
        "oxy_euler_frequency_data.txt",
        "oxy_verlet_frequency_data.txt",
        "nit_euler_frequency_data.txt",
        "nit_verlet_frequency_data.txt"]
    frequency_data0 = []
    frequency_data1 = []
    frequency_data2 = []
    frequency_data3 = []
    dt0 = []
    dt1 = []
    dt2 = []
    dt3 = []
    energy_data0 = []
    energy_data1 = []
    energy_data2 = []
    energy_data3 = []
    for i in range(len(files)):
        infile = open(files[i], "r")
        maximum_line = []
        for line in infile.readlines():
            tokens = line.split(",")
            if i == 0:
                dt0.append(float(tokens[0]))
                frequency_data0.append(float(tokens[1]))
                energy_data0.append(float(tokens[2]))
            elif i == 1:
                dt1.append(float(tokens[0]))
                frequency_data1.append(float(tokens[1]))
                energy_data1.append(float(tokens[2]))
            elif i == 2:
                dt2.append(float(tokens[0]))
                frequency_data2.append(float(tokens[1]))
                energy_data2.append(float(tokens[2]))
            elif i == 3:
                dt3.append(float(tokens[0]))
                frequency_data3.append(float(tokens[1]))
                energy_data3.append(float(tokens[2]))
            maximum_line.append(0.5)
        infile.close()

    fig1, ax1 = pyplot.subplots()
    ax1.set_xlim(0, 0.2)
    ax1.set_ylim(0, 2)
    pyplot.plot(dt0, frequency_data0, label="Frequency Inaccuracy data Oxygen Molecule for Euler sim")
    pyplot.plot(dt1, frequency_data1, label="Frequency Inaccuracy data Oxygen Molecule for Verlet sim")
    pyplot.plot(dt2, frequency_data2, label="Frequency Inaccuracy data Nitrogen Molecule for Euler sim")
    pyplot.plot(dt3, frequency_data3, label="Frequency Inaccuracy data Nitrogen Molecule for Verlet sim")
    pyplot.hlines(maximum_line, xmin=0, xmax=dt0[-1], colors='r',
                  linestyles="dotted",
                  label="Maximum inaccuracy before data is excluded")
    pyplot.xlabel(f"Time step, T = {time_units} s")
    pyplot.ylabel("Frequency Inaccuracy, %")
    pyplot.title("Plot of frequency inaccuracy against time step.")
    pyplot.legend()
    fig1.set_size_inches(15, 8)
    ax1.annotate("Max dt = 0.12 T", (0.12, 0.443), xytext=(0.15, 0.25), arrowprops=dict(shrink=0.05))
    ax1.annotate("Max dt = 0.09 T", (0.09, 0.27), xytext=(0.09, 0.05), arrowprops=dict(shrink=0.05))
    ax1.annotate("Max dt = 0.08 T", (0.08, 0.443), xytext=(0.08, 1), arrowprops=dict(shrink=0.05))
    ax1.annotate("Max dt = 0.07 T", (0.07, 0.445), xytext=(0.03, 0.75), arrowprops=dict(shrink=0.05))

    fig2, ax2 = pyplot.subplots()
    pyplot.plot(dt0, energy_data0, label="Energy inaccuracy data for Oxygen Molecule for Euler sim")
    pyplot.plot(dt1, energy_data1, label="Energy inaccuracy data for Oxygen Molecule for Verlet sim")
    pyplot.plot(dt2, energy_data2, label="Energy inaccuracy data for Nitrogen Molecule for Euler sim")
    pyplot.plot(dt3, energy_data3, label="Energy inaccuracy data for Nitrogen Molecule for Verlet sim")
    pyplot.xlabel(f"Time step, T = {time_units} s")
    pyplot.ylabel("Energy Inaccuracy, %")
    pyplot.title("Plot of Energy inaccuracy against time step.")
    pyplot.legend()
    fig2.set_size_inches(15,8)

    fig1.show()
    fig2.show()
    fig1.savefig("frequency_inaccuracy_plot")
    fig2.savefig("energy_inaccuracy_plot")


main()

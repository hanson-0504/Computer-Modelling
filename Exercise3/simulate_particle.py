"""
Symplectic Euler and Velocity Verlet time integration of
a particle moving in a double well potential.

Produces plots of the position of the particle
and its energy, both as function of time. Also
saves both to file.

The potential is V(x) = a*x^4 - b*x^2, where
a and b are hard-coded in the main() method
and passed to the functions that
calculate force and potential energy.

This has been ammended to simulate two particles
following a Morse Potential. Using Symplectic Euler
and Velocity Verlet integration methods.

Morse Potential is given by
U(r_1, r_2) = D_e*[-1 + (1-e**(-alpha*(r12 - r_e)))**2]
where D_e, alpha and r_e are read into using text files.

This simulation produces the wavenumber of oscillation,
period and frequency of oscillation,
including the energy and frequency inaccuracy measurements.

Author: Sam Hanson
Number: s2153833

"""

import sys
import math
import numpy as np
import scipy.signal as ss
import scipy.constants as sc
import matplotlib.pyplot as pyplot
from particle3D import Particle3D


def force_double_well(p1, a, b):
    """
    Return the force on a particle in a double well potential.

    The force is given by
        F(x) = -dV/dx = -4*a*x^3 + 2*b*x

    Parameters
    ----------
    p1: Particle1D
    a: float
    b: float

    Returns
    -------
    force: float
    """
    force = -4 * a * p1.position**3 + 2 * b * p1.position
    return force


def morse_force(p1, p2, a, D, r):
    """
    Return the force on a particle in a morse potential.

    The force is given by
        F(r1, r2) = -dU/dx = 2*a*D*((1-e**(-a*(r12-r)))**2)*(e**(-a*(r12-r)))*r12/mag(r12)
    Parameters
    ----------
    p1 : Particle3D
    p2 : Particle3D
    a : float
    D : float
    r : float

    Returns
    -------
    force : Force on p1 from p2 in a Morse potential:
        Numpy array

    """
    r12 = p2.position - p1.position  # Vector from p2 to p1
    mag12 = math.sqrt(np.dot(r12, r12))  # distance between p1 and p2
    norm12 = r12/mag12  # Normalised vector
    force = 2*a*D*norm12*(1 - math.exp(-a*(mag12 - r)))*(math.exp(-a*(mag12 - r)))
    return force


def potential_double_well(p1, a, b):
    """
    Method to return potential energy
    of particle in double-well potential
    V(x) = a*x^4 - b*x^2

    Parameters
    -----------
    p1: Particle1D
    a: float
    b: float

    Returns
    -------
    potential: float
    """
    potential = a * p1.position**4 - b * p1.position**2
    return potential


def potential_morse(p1, p2, a, D, r):
    """
    Returns morse potential energy between p1 and p2

    Potential is given by
        U(r1, r2) = D*[-1 + (1-e**(-a*(r12 - r)))**2]

    Parameters
    ----------
    p1 : Particle3D
    p2 : Particle3D
    a : float
    D : float
    r : float

    Returns
    -------
    potential : Morse potential on p1 and p2
        float

    """
    r12 = p2.position - p1.position  # vector from p2 to p1
    mag12 = math.sqrt(np.dot(r12, r12))  # distance between p1 and p2
    potential = D*(-1 + (1 - math.exp(-a*(mag12 - r)))**2)
    return potential


def calc_period(positions, time_units, dt):
    """
    Calculates the period using scipy.signal.find_peaks.

    Parameters
    ----------
    positions : List of all the distances between p1 and p2. (list)
    time_units : Converts program time units to seconds. (float)

    Returns
    -------
    period : Average time period between peaks of position plot. (float)
    """
    # finds all peaks in positions list and takes the time coordinate
    peak, _ = ss.find_peaks(positions)
    # changing array to numpy array
    peak = np.array(peak)
    diff_list = []
    for i in range(1, len(peak)):
        # time difference between peak[i - 1] and peak[i] for all peaks
        diff_list.append(peak[i] - peak[i - 1])
    # averaging the list of time differences
    period = np.average(diff_list)
    # converting units to seconds
    period = period*time_units*dt
    print(f"Period = {period:.03} s")
    print(f"Frequency = {1/period:.03} Hz")
    return period


def wavenumber(period):
    """
    Calculates the wavenumber of oscillation using oscillation period

    Parameters
    ----------
    period : Oscillation period (float)

    Returns
    -------
    v : Vibrational frequency (float)
        v = 1/cT , T is period, c is speed of light

    """
    freq = 1/period
    v = freq/sc.c  # c = speed of light in a vacuum, v is in 1/m
    v = v*0.01  # 1/m to 1/cm
    return v


def frequency_accuracy(mode, test, v):
    """
    Calculate the accuracy of the frequency measurement, using a frequency
    calculated with dt = 1e-5.

    Parameters
    ----------
    mode : Euler or Verlet integration modes. str
    test : Input file for the 4 tests
        test1_oxygen.txt
        test1_nitrogen.txt
        test2_oxygen.txt
        test2_nitrogen.txt
    v : Frequency measurement at current dt

    Returns
    -------
    delta : Change in frequency between frequency at current dt and dt = 1e-5

    """
    if mode == "euler":
        if test == "test1_oxygen.txt":
            v0 = 45735156621870.38
        elif test == "test1_nitrogen.txt":
            v0 = 68563469984165.64
        elif test == "test2_oxygen.txt":
            v0 = 41497230094680.07
        elif test == "test2_nitrogen.txt":
            v0 = 65621183752027.78
    elif mode == "verlet":
        if test == "test1_oxygen.txt":
            v0 = 45735175980904.47
        elif test == "test1_nitrogen.txt":
            v0 = 68563498136387.71
        elif test == "test2_oxygen.txt":
            v0 = 41497230094680.07
        elif test == "test2_nitrogen.txt":
            v0 = 65621183752027.78
    delta = abs(v - v0)/v0
    return delta


def energy_accuracy(energies, mode):
    """
    Calculate the accuracy in the energy measurement.

    Parameters
    ----------
    energies : list of all energy measurement

    Returns
    -------
    accuracy : accuracy of energy measurements

    """
    init_energy = energies[0]  # initial energy measurement
    # Find E_max
    e_max = np.max(energies)
    e_min = np.min(energies)
    # Delta E = (E_max - E_min)/E_0
    accuracy = (abs(e_max) - abs(e_min))/init_energy
    return accuracy


def initialise(filename):
    """
    Acquires initial data from text files

    Parameters
    ----------
    filename : Filename of the text file which has the initail data in the
        form of:
            Oxygen/Nitrogen
            D
            r
            a
            m
            x1
            x2
            x3
            v1
            v2
            v3

    Returns
    -------
    D : D float from exercise3 task
    r : r float from exercise3 task
    a : alpha float from exercise 3 task
    m : mass of particle, float
    x : array of x-position, y-position, z-position, floats
    v : array of x-velocity, y-velocity, z-velocity, floats

    """
    inputdata = []
    filein = open(filename, "r")
    for line in filein.readlines():
        if not line.startswith("#"):
            inputdata.append(line)
    filein.close()
    D = float(inputdata[1])
    r = float(inputdata[2])
    a = float(inputdata[3])
    m = float(inputdata[4])
    x = np.array((float(inputdata[5]), float(inputdata[6]), float(inputdata[7])))
    v = np.array((float(inputdata[8]), float(inputdata[9]), float(inputdata[10])))
    return D, r, a, m, x, v


def main():
    # Read inputs from command line
    # The variable sys.argv contains whatever you typed on the command line
    # or after %run on the ipython console to launch the code.  We can use
    # it to get user inputs.
    # Here we expect three things:
    #    the name of this file
    #    euler or verlet
    #    the name of the output file the user wants to write to
    # So we start by checking that all three are specified and quit if not,
    # after giving the user a helpful error message.
    if len(sys.argv) != 5:
        print("You left out inputs when running.")
        print("In spyder, run like this instead:")
        print(f"    %run {sys.argv[0]} <euler or verlet> <desired output file> <desired input file> <dt>")
        sys.exit(1)
    else:
        mode = sys.argv[1]
        outfile_name = sys.argv[2]
        infile_name = sys.argv[3]
        dt = float(sys.argv[4])
        # Euler/O2 = 0.12
        # Euler/N2 = 0.09
        # verlet/O2 = 0.08
        # Verlet/N2 = 0.07

    # Open the output file for writing ("w")
    outfile = open(outfile_name, "w")

    # Choose our simulation parameters
    # dt is given from the command line
    numstep = int(25/dt)
    time = 0.0

    # Set up particle initial conditions:
    D, r, a, m, x, v = initialise(infile_name)

    p1 = Particle3D("p1", m, x, v)
    p2 = Particle3D("p2", m, -x, -v)

    # Get initial force
    force = morse_force(p1, p2, a, D, r)

    # Write out starting time, position, and energy values
    # to the output file.
    energy = p1.kinetic_energy() + potential_morse(p1, p2, a, D, r) + p2.kinetic_energy()
    outfile.write(f"{time}    {p1.position}    {p2.position}    {energy}\n")

    # Initialise numpy arrays that we will plot later, to record
    # the trajectories of the particles.
    times = np.zeros(numstep)
    positions = np.zeros(numstep)
    energies = np.zeros(numstep)

    # Start the time integration loop
    for i in range(numstep):

        # Update the positions and velocities.
        # This will depend on whether we are doing an Euler or verlet integration
        if mode == "euler":
            # Update particle position
            p1.update_position_1st(dt)
            p2.update_position_1st(dt)

            # Calculate force
            force = morse_force(p1, p2, a, D, r)

            # Update particle velocity
            p1.update_velocity(dt, force)
            p2.update_velocity(dt, (-1)*force)

        elif mode == "verlet":
            # Update particle position using previous force
            p1.update_position_2nd(dt, force)
            p2.update_position_2nd(dt, (-1)*force)

            # Get the force value for the new positions
            force_new = morse_force(p1, p2, a, D, r)

            # Update particle velocity by averaging
            # current and new forces
            p1.update_velocity(dt, 0.5*(force + force_new))
            p2.update_velocity(dt, -0.5*(force + force_new))

            # Re-define force value for the next iteration
            force = force_new
        else:
            raise ValueError(f"Unknown mode {mode} - should be euler or verlet")

        # Increase time
        time += dt

        # Output particle information
        energy = p1.kinetic_energy() + potential_morse(p1, p2, a, D, r) + p2.kinetic_energy()
        outfile.write(f"{time} {p1.position}   {p2.position} {energy}\n")

        # Store the things we want to plot later in our arrays
        times[i] = time
        positions[i] = np.linalg.norm(p1.position - p2.position)
        energies[i] = energy

    # Now the simulation has finished we can close our output file
    outfile.close()

    # time_units converts the "time" in the code to seconds
    time_units = (
        sc.value("Angstrom star"))*math.sqrt(
                 sc.value("atomic mass constant")/sc.value("electron volt"))
    period = calc_period(positions, time_units, dt)
    print(f"Wavenumber = {round(wavenumber(period), 0)} 1/cm")

    # Energy inaccuarcy
    print(f"Energy inaccuracy = {100*energy_accuracy(energies, mode):.03} %")
    # Frequency inaccuracy
    frequency_inaccuracy = frequency_accuracy(mode, infile_name, 1/period)
    print(f"Frequency inaccuracy = {100*frequency_inaccuracy:.03} %")
    if frequency_accuracy(mode, infile_name, (1/period)) <= 0.5/100:
        print("This timestep is a good approximation!")
    else:
        print("This timestep is a bad approximation!")

    # Plot particle trajectory to screen. There are no units
    # here because it is an abstract simulation, but you should
    # include units in your plot labels!
    if mode == "euler":
        pyplot.figure()
        pyplot.title(f'Symplectic Euler: position vs time, dt = {dt} t')
        # Units = 10**-10 * sqrt(amu / eV) = time_units
        pyplot.xlabel(f'Time (t = {time_units} s)')
        # Position units = 10**-10 m
        pyplot.ylabel('Position (10**-10 m)')
        pyplot.plot(times, positions)
        pyplot.show()

        # Plot particle energy to screen
        pyplot.figure()
        pyplot.title(f'Symplectic Euler: total energy vs time, dt = {dt} t')
        # Units = 10**-10 * sqrt(amu / eV) = time_units
        pyplot.xlabel(f'Time (t = {time_units} s)')
        # Energy units = electron volts
        pyplot.ylabel('Energy (eV)')
        pyplot.plot(times, energies)
        pyplot.show()

    elif mode == "verlet":
        pyplot.figure()
        pyplot.title(f'Verlet: position vs time, dt = {dt} t')
        # Units = 10**-10 * sqrt(amu / eV) = time_units
        pyplot.xlabel(f'Time (t = {time_units:.03} s)')
        # Position units = 10**-10 m
        pyplot.ylabel('Position (10**-10 m)')
        pyplot.plot(times, positions)
        pyplot.show()

        # Plot particle energy to screen
        pyplot.figure()
        pyplot.title(f'Verlet: total energy vs time, dt = {dt} t')
        # Units = 10**-10 * sqrt(amu / eV) = time_units
        pyplot.xlabel(f'Time (t = {time_units} s)')
        # Energy units = electron volts
        pyplot.ylabel('Energy (eV)')
        pyplot.plot(times, energies)
        pyplot.show()

    else:
        raise ValueError(f"Unknown mode {mode} - should be euler or verlet")

# This strange but standard python idiom means that the main function will
# only be run if we run this file, not if we just import it from another
# python file. It is good practice to include it whenever your code can be
# run as a program.
if __name__ == "__main__":
    main()


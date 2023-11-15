"""
CompMod Ex2: Particle3D, a class to describe point particles in 3D space

An instance describes a particle in Euclidean 3D space:
velocity and position are [3] arrays

Author: Sam Hanson
Number: s2153833

"""
import numpy as np


class Particle3D(object):
    """
    Class to describe point-particles in 3D space

    Attributes
    ----------
    label: name of the particle
    mass: mass of the particle
    position: position of the particle
    velocity: velocity of the particle

    Methods
    -------
    __init__
    __str__
    kinetic_energy: computes the kinetic energy
    momentum: computes the linear momentum
    update_position_1st: updates the position to 1st order
    update_position_2nd: updates the position to 2nd order
    update_velocity: updates the velocity

    Static Methods
    --------------
    read_file: initializes a P3D instance from a file handle
    total_kinetic_energy: computes total K.E. of a list of particles
    com_velocity: computes centre-of-mass velocity of a list of particles
    """

    def __init__(self, label, mass, position, velocity):
        """
        Initialises a particle in 3D space.

        Parameters
        ----------
        label: str
            name of the particle
        mass: float
            mass of the particle
        position: [3] float array
            position vector
        velocity: [3] float array
            velocity vector
        """
        self.label = label
        self.mass = mass
        self.position = np.array(position)
        self.velocity = np.array(velocity)

    def __str__(self):
        """
        Return an XYZ-format string. The format is
        label    x  y  z

        Returns
        -------
        str
        """
        a = str(f"{self.label}   {self.position[0]}  {self.position[1]} {self.position[2]}")
        return a

    def kinetic_energy(self):
        """
        Returns the kinetic energy of a Particle3D instance

        Returns
        -------
        ke: float
            1/2 m v**2
        """
        v = np.linalg.norm(self.velocity)
        ke = 0.5*(self.mass)*v**2
        return ke

    def momentum(self):
        """
        Returns the momentum of a Particle 3D instance

        Returns
        -------
        p: numpy array
            m v
        """
        p = self.mass*self.velocity

        return p

    def update_position_1st(self, dt):
        """
        Returns position of Particle 3D instance at t = t + dt

        Returns
        -------
        self.position: numpy array
            r(t + dt) = r(t) + dt v(t)
        """
        self.position = self.position + dt*self.velocity

    def update_position_2nd(self, dt, force):
        """
        Returns position of Particle 3D instance at t = t + dt
        2nd order integration method.

        Returns
        -------
        self.position: numpy array
            r(t+dt) = r(t) + dt v(t) + dt^2 f(t)/2m
        """
        self.position = self.position + dt*self.velocity + (dt**2)*(
            force/(2*self.mass))

    def update_velocity(self, dt, force):
        """
        Returns the velocity of Particle 3D instance at t = t + dt

        Returns
        -------
        self.velocity: numpy array
            v(t + dt) = v(t) + dt f/m
        """
        self.velocity = self.velocity + dt*force/self.mass

    @staticmethod
    def read_line(line):
        """
        Creates a Particle3D instance given a line of text.

        The input line should be in the format:
        label   <mass>  <x> <y> <z>    <vx> <vy> <vz>

        Parameters
        ----------
        filename: str
            Readable file handle in the above format

        Returns
        -------
        p: Particle3D
        """
        tokens = line.split()
        label = str(tokens[0])
        mass = float(tokens[1])
        position = [float(tokens[2]), float(tokens[3]), float(tokens[4])]
        velocity = [float(tokens[5]), float(tokens[6]), float(tokens[7])]
        p = Particle3D(label, mass, position, velocity)
        return p

    @staticmethod
    def total_kinetic_energy(particles):
        """
        Returns sum of all Particle 3D objects kinetic energies.

        Returns
        -------
        total_ke: float
            total_ke = ke[0] + ke[1] + ke[2] +...
        """
        total_ke = 0
        for i in range(len(particles)):
            total_ke += particles[i].kinetic_energy()
        return total_ke

    @staticmethod
    def com_velocity(particles):
        """
        Computes the CoM velocity of a list of P3D's

        Parameters
        ----------
        particles: list
            A list of Particle3D instances

        Returns
        -------
        com_vel: array
            Centre-of-mass velocity
        """
        # v_com = (sum of all particle momentum / total mass)
        total_mass = 0
        momentum_com = 0
        for i in range(len(particles)):
            total_mass += particles[i].mass
            momentum_com += particles[i].momentum
        com_vel = momentum_com / total_mass
        return com_vel

# Fireworks - N-Body simulations Python library

Fireworks is a Python package developed in the context of the Computational Astrophysics course at the University of Padova (2023-2024). 

It contains all the functions needed to run N-Body simulations. This repository contains:

* Docker image with the required packages.
* `Fireworks` package: install it by running `pip install fireworks`
* `project` folder: here you can find the final group project for the exam. It consists in three parts: CPU optimization, GPU optimization, Plummer Sphere simulation. I personally worked on the CPU optimization part.

## How to use Fireworks

After running `pip install fireworks` you can start running your first N-Body simulations. All functions are well documented, so take a look at them. A brief recap:

* `fireworks.ic` : This module contains functions and utilities to generate initial conditions for the Nbody simulations. Each function/class returns an instance of the class :class:`~fireworks.particles.Particles`

* `fireworks.particles` : Main class used to store Nbody particles data.

* `fireworks.nbodylib`: It contains the core functions needed to carry on a N-Body simulation:

    * `fireworks.nbodylib.dynamics` : This module contains a collection of functions to estimate acceleration due to gravitational forces.

    * `fireworks.nbodylib.dynamics`
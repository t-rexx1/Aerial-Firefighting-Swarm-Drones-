"""
ME 144/244: Modeling, Simulation, and Digital Twins of Drone-Based Systems
Project 7: Aerial Firefighting
Spring 2026 Semester

"""
import numpy as np
from ga_class import GeneticAlgorithm as _GeneticAlgorithmBase
from simulation import Simulation
"""
This script features the function to run the genetic algorithm.
It calls the simulation function from the Simulation class to evaluate the cost of each design.

LAM vector:
0. magnitude of plane velocity          : [50, 100] m/s
1-3. initial plane position (x0,y0,z0)  : [(-100,100,-100), (100,250,100)] m # NOTE: x0 is fixed at zero, y0 range is [50, 250]
4. particle radius                      : [0.05, 0.2] m  # NOTE: physically, retardant droplets are 0.0003-0.003 m in diameter.
#   These coarse radii are a deliberate simplification to keep particle counts tractable
#   (realistic droplet sizes would require billions of particles for a 10 m^3 tank).
5. sprayer amplitude                    : [0, 0.5]
6. start of release time                : [0, 0.25]*(total flight time)
7. end of release time                  : [0.25, 1.0]*(total flight time)
8. drop rate of retardant               : [2, 10] * m^3 / s
9. particle drop velocity               : [0, 150] * m / s
10. amplitude of parametric curve       : [1, 100]
11. wavelength of parametric curve      : [500, 2000]
"""


class GeneticAlgorithm(_GeneticAlgorithmBase):

    def __init__(self, parameters):
        super().__init__(parameters)
        self.initialize_special_params()

    def initialize_special_params(self):
        # Map uppercase parameter names to names expected by base class
        self.numLam = self.NUMLAM
        self.tolerance = self.TOLERANCE

    def _generate_design_string(self):
        rng = np.random.default_rng()
        LAM = np.array([
            rng.uniform(50, 100), # aircraft speed in m/s
            0., # x0 fixed at zero so that the plane starts at one end of the domain and not in the middle, give z the freedom
            rng.uniform(self.Y0_MIN, self.Y0_MAX), # y0 ranges from 50 to 250 meters above ground
            rng.uniform(-100, 100), # z0 ranges from -100 to 100 m
            # rng.uniform(0.05, 0.2), # particles are coarse for simulation approximation, parallelization would help
            rng.uniform(0.001, 0.01), # particles are coarse for simulation approximation, parallelization would help
            rng.uniform(0, 0.5), # controls how wide the "cone" of particles is when they are released from the plane
            rng.uniform(0, 0.1) , # we can start spraying anywhere from the beginning to 1/4 of the way into the flight
            rng.uniform(0.25, 1) , # we can stop spraying anywhere from 1/4 to the end of the flight 
            rng.uniform(2, 10), # volume of retardant dropped per second
            rng.uniform(0, 150), # particle drop velocity 
            rng.uniform(1, 100), # amplitude of parametric curve
            rng.uniform(500, 2000) # wavelength of parametric curve
        ])
        return LAM

    def _evaluate_costs(self, start):
        for i in range(start, self.S):
            sim = Simulation(self.parameters, LAM=self.strings[i])
            _, _ = sim.simulate_path_with_nozzles()
            self.costs_of_current_generation[i] = sim.calculate_cost()

    def specialGA(self):
        raise NotImplementedError


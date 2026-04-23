"""
ME 144/244: Modeling, Simulation, and Digital Twins of Drone-Based Systems
Project 7: Aerial Firefighting
Spring 2026 Semester

"""
import numpy as np
from abc import ABC, abstractmethod
from typing import List
import pickle
from animation import Animation

"""
This script runs the simulation for the aerial firefighting project.
Fill in every blank (marked with a # comment) to complete the simulation.
"""


class Aircraft:
    def __init__(self, LAM: List[int], spacing):
        self.spacing = spacing  # meters
        self.speed = LAM[0]
        self.x0 = LAM[1]
        self.y0 = LAM[2]
        self.z0 = LAM[3]
        self.pos = np.array([self.x0, self.y0, self.z0])
        self.dir = np.array([1., 0., 0.])
        self.vel = self.speed * self.dir


class Curve(ABC):
    @abstractmethod
    def get_position(self, u):
        pass

    @abstractmethod
    def get_tangent(self, u):
        pass


class BezierCurve(Curve):
    def __init__(self, control_points):
        self.control_points = np.array(control_points)

    def get_position(self, u):
        points = self.control_points.copy()
        n = len(points)
        for r in range(1, n):
            for i in range(n - r):
                points[i] = (1 - u) * points[i] + u * points[i + 1]
        return points[0]

    def get_tangent(self, u):
        points = self.control_points.copy()
        n = len(points)
        diff_points = (n - 1) * (points[1:] - points[:-1])
        for r in range(1, len(diff_points)):
            for i in range(len(diff_points) - r):
                diff_points[i] = (1 - u) * diff_points[i] + u * diff_points[i + 1]
        tangent = diff_points[0]
        return tangent / np.linalg.norm(tangent)


class SineWave(Curve):
    def __init__(self, amplitude, wavelength, x0, y0, z0):
        self.amplitude = amplitude
        self.wavelength = wavelength
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0

    def get_position(self, u):
        k = 2 * np.pi / self.wavelength
        x = u + self.x0
        y = self.y0
        z = self.amplitude * np.sin(k * u) + self.z0
        return np.array([x, y, z])

    def get_tangent(self, u):
        k = 2 * np.pi / self.wavelength
        dx_du = 1.0
        dy_du = 0.0
        dz_du = self.amplitude * k * np.cos(k * u)
        tangent = np.array([dx_du, dy_du, dz_du])
        return tangent / np.linalg.norm(tangent)


class Simulation:

    def __init__(self, parameters, LAM):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.u = 0.0
        self.G = np.array([self.GX, self.GY, self.GZ])

        self.history_plane = []
        self.history_particles = []

        self.r_particle = LAM[4]
        self.spray_amplitude = LAM[5]
        self.release_time_start = LAM[6]
        self.release_time_end = LAM[7]
        self.flow_rate = LAM[8]
        self.eject_speed = LAM[9]
        self.curve_amplitude = LAM[10]
        self.curve_wavelength = LAM[11]

        self.vol_particle = 4/3 * np.pi * self.r_particle**3
        self.num_particles_per_second = self.flow_rate / self.vol_particle * self.FRACTION_PARTICLES_TRACKED
        self.num_particles = int(self.TANK_VOLUME / self.vol_particle * self.FRACTION_PARTICLES_TRACKED) # Increased for 4 nozzles
        self.r_drop = np.full((self.num_particles, 3), np.nan)
        self.v_drop = np.zeros((self.num_particles, 3))
        self.active = np.zeros(self.num_particles, dtype=bool)
        self.landed = np.zeros(self.num_particles, dtype=bool)
        self.drops_released = 0
        self.airplane = Aircraft(LAM, self.SPACING)


    # --- Step 1: Update aircraft state ---
    # The aircraft travels along a parametric curve at constant airspeed V.
    # Each timestep we advance the curve parameter u by du = V*DT / ||tangent||,
    # which ensures the aircraft moves exactly V*DT meters along the curve.
    def _update_aircraft_state(self, airplane, curve):
        pos     = # get position on curve at current self.u
        tangent = # get tangent on curve at current self.u

        tangent_norm = np.linalg.norm(tangent)
        du       = # how far to advance u this timestep: (speed * DT) / tangent_norm
        self.u  += du

        airplane.pos = # update position to pos
        airplane.dir = tangent / tangent_norm
        airplane.vel = airplane.dir * airplane.speed

        global_up   = np.array([0., 1., 0.])
        plane_right = np.cross(airplane.dir, global_up)
        plane_right /= np.linalg.norm(plane_right)

        return plane_right


    # --- Step 2: Compute nozzle positions ---
    # The aircraft has 4 nozzles. Their base positions cycle as nozzle_idx % 4:
    #   0 — center   (NOZZLE_AFT_OFFSET m behind CG, no lateral shift)
    #   1 — right    (same aft offset, shifted +spacing in plane_right)
    #   2 — left     (same aft offset, shifted -spacing in plane_right)
    #   3 — extra aft (shifted -spacing further along the aircraft direction)
    # The sprayer end is NOZZLE_DROP_OFFSET m below the base in the local-up direction.
    def _compute_nozzle_positions(self, airplane, plane_right, count):
        plane_up   = np.cross(plane_right, airplane.dir)   # local up unit vector
        aft_offset = -self.NOZZLE_AFT_OFFSET * airplane.dir

        nozzle_idx = np.arange(self.drops_released, self.drops_released + count) % 4

        base_positions = np.tile(airplane.pos + aft_offset, (count, 1))
        base_positions[nozzle_idx == 1] += # right nozzle: +spacing in plane_right direction
        base_positions[nozzle_idx == 2] -= # left nozzle:  -spacing in plane_right direction
        base_positions[nozzle_idx == 3] -= # aft nozzle:   -spacing along airplane.dir direction

        end_positions = base_positions - # NOZZLE_DROP_OFFSET below base in plane_up direction

        return base_positions, end_positions


    # --- Step 3: Compute initial droplet velocities ---
    # The spray direction is derived from the nozzle geometry, then randomly perturbed
    # to simulate the spread of the spray cone.
    def _compute_spray_velocities(self, airplane, plane_right, sprayer_base, sprayer_end, count):
        # Unit vector pointing from sprayer base to end (downward along nozzle axis)
        diff   = sprayer_end - sprayer_base
        norms  = np.linalg.norm(diff, axis=1, keepdims=True)
        n_base = diff / norms

        # Random unit vector — gives each droplet a slightly different direction
        n_rand  = np.random.uniform(-1, 1, (count, 3))
        n_rand /= np.linalg.norm(n_rand, axis=1, keepdims=True)

        # Perturb the nozzle direction by spray_amplitude and re-normalise
        directions = # n_base + spray_amplitude * n_rand
        n_spray    = directions / np.linalg.norm(directions, axis=1, keepdims=True)

        # All droplets inherit the aircraft velocity at the moment of release
        v_nozzle = np.tile(airplane.vel, (count, 1))

        return # v_nozzle + eject_speed * n_spray


    # --- Step 4: Release droplets ---
    # Droplets are only released during [release_time_start, release_time_end).
    # The number released per timestep is floor(flow_rate * DT / vol_particle).
    def _release_droplets(self, airplane, plane_right, t):
        time_fraction = t / self.STEPS
        if (self.drops_released >= self.num_particles
                or time_fraction < self.release_time_start
                or time_fraction >= self.release_time_end):
            return

        count = min(
            int(self.num_particles_per_second * self.DT),
            self.num_particles - self.drops_released
        )
        if count == 0:
            return

        new_indices  = np.arange(self.drops_released, self.drops_released + count)
        sprayer_base, sprayer_end = self._compute_nozzle_positions(airplane, plane_right, count)
        velocities   = self._compute_spray_velocities(airplane, plane_right, sprayer_base, sprayer_end, count)

        # Pre-integrate one timestep so droplets start one step ahead of the nozzle
        positions = sprayer_end + # DT * velocities

        self.r_drop[new_indices]  = positions
        self.v_drop[new_indices]  = velocities
        self.active[new_indices]  = True
        self.drops_released      += count


    # --- Step 5: Fire zone mask ---
    # Returns True for each particle that is inside the fire footprint.
    # Two shapes are supported via self.FIRE_ZONE_SHAPE: 'box' or 'circle'.
    def _in_fire_zone(self, r_xz):
        shape = getattr(self, 'FIRE_ZONE_SHAPE', 'box')

        if shape == 'circle':
            cx = getattr(self, 'FIRE_CENTER_X', (self.FIRE_X_MIN + self.FIRE_X_MAX) / 2)
            cz = getattr(self, 'FIRE_CENTER_Z', (self.FIRE_Z_MIN + self.FIRE_Z_MAX) / 2)
            r  = getattr(self, 'FIRE_RADIUS',
                         min(self.FIRE_X_MAX - self.FIRE_X_MIN,
                             self.FIRE_Z_MAX - self.FIRE_Z_MIN) / 2)
            dist2 = # (x - cx)^2 + (z - cz)^2
                    # hint: r_xz[:, 0] is x, r_xz[:, 1] is z
            return dist2 <= # r ** 2
        else:  # 'box'
            return (
                (r_xz[:, 0] > self.FIRE_X_MIN) & (r_xz[:, 0] < self.FIRE_X_MAX) &
                (r_xz[:, 1] > self.FIRE_Z_MIN) & (r_xz[:, 1] < self.FIRE_Z_MAX)
            )


    # --- Step 6: Compute fire updraft ---
    # Above the fire zone, hot air rises. The updraft speed decays exponentially
    # with height: v_up(y) = UPDRAFT_VELOCITY * exp(-UPDRAFT_DECAY * y).
    # It acts only in the vertical (y) direction.
    def _compute_updraft(self, r_curr):
        v_up    = np.zeros_like(r_curr)
        in_fire = self._in_fire_zone(r_curr[:, [0, 2]])

        if np.any(in_fire):
            y_heights          = r_curr[in_fire, 1]
            v_up[in_fire, 1]  = # UPDRAFT_VELOCITY * exp(-UPDRAFT_DECAY * max(0, y))
                                 # hint: use np.exp and np.maximum(0, y_heights)

        return v_up


    # --- Step 7: Compute aerodynamic drag ---
    # Drag depends on the Reynolds number Re. We use the Chow piecewise Cd model.
    # The drag force vector points in the same direction as the relative velocity
    # (fluid pushes particle in the direction the fluid is moving relative to the particle).
    def _compute_drag(self, v_rel):
        speed      = np.linalg.norm(v_rel, axis=1)
        safe_speed = np.where(speed == 0, 1e-10, speed)  # avoid division by zero

        Re = # Reynolds number: 2 * r_particle * RHO_A * safe_speed / MU_F

        # Chow piecewise drag coefficient — numpy handles the array automatically
        Cd = np.where(Re <= 1,    24.0 / Re,
             np.where(Re <= 400,  24.0 / Re**0.646,
             np.where(Re <= 3e5,  0.5,
             np.where(Re <= 2e6,  0.000366 * Re**0.4275,
                                  0.18))))

        A_i    = np.pi * self.r_particle**2
        f_drag = (0.5 * self.RHO_A * A_i
                * Cd[:, np.newaxis]
                * safe_speed[:, np.newaxis]
                * v_rel)

        return f_drag


    # --- Step 8: Step particle physics (Forward Euler) ---
    # Each timestep: compute forces, update velocity, update position.
    # Newton's 2nd law: m*a = f_drag + m*g  =>  a = f_drag/m + g
    def _step_physics(self, flying_mask):
        if not np.any(flying_mask):
            return

        r_curr = self.r_drop[flying_mask]
        v_curr = self.v_drop[flying_mask]

        v_wind = np.array([self.WIND_X, self.WIND_Y, self.WIND_Z])
        v_up   = self._compute_updraft(r_curr)

        # Relative velocity of the fluid with respect to the droplet
        v_rel  = # (v_wind + v_up) - v_curr

        f_drag = self._compute_drag(v_rel)

        # Acceleration = drag force / particle mass + gravity
        # particle mass = RETARDANT_DENSITY * vol_particle
        acc = # f_drag / (RETARDANT_DENSITY * vol_particle) + G

        # Forward Euler integration
        v_curr += # acc * DT
        r_curr += # v_curr * DT

        # Ground collision: clamp y to 0 and mark particle as landed
        hits = r_curr[:, 1] <= 0
        if np.any(hits):
            r_curr[hits, 1] = 0.0

        self.r_drop[flying_mask]  = r_curr
        self.v_drop[flying_mask]  = v_curr
        self.landed[flying_mask]  = self.landed[flying_mask] | hits


    # --- Step 9: Main simulation loop ---
    # Ties everything together: advance the aircraft, release droplets each timestep,
    # then step the physics for all airborne particles.
    def simulate_path_with_nozzles(self, seed=144):
        np.random.seed(seed)
        airplane   = self.airplane
        plane_curve = SineWave(
            self.curve_amplitude, self.curve_wavelength,
            airplane.x0, airplane.y0, airplane.z0
        )

        for t in range(self.STEPS):
            plane_right = self._update_aircraft_state(airplane, plane_curve)
            self._release_droplets(airplane, plane_right, t)
            self._step_physics(# mask for particles that are active AND not yet landed)
            self.history_plane.append(airplane.pos.copy())
            self.history_particles.append(self.r_drop.copy())
            if self.drops_released > 0 and self.landed.sum() == self.drops_released:
                break

        return self.history_plane, self.history_particles


    # --- Step 10: Compute cost ---
    # Lower cost = better. We want as many droplets in the fire zone as possible,
    # and we want the aircraft to fly as high as possible (pilot safety).
    def calculate_cost(self):
        landed_xz = self.r_drop[self.landed][:, [0, 2]]
        in_fire   = self._in_fire_zone(landed_xz)
        N_hit     = sum(in_fire)
        N_total   = self.num_particles
        return # W_1 * (1 - N_hit / N_total) + W_2 * (1 - airplane.y0 / Y0_MAX)
               # hint: use self.W_1, self.W_2, self.airplane.y0, self.Y0_MAX


# Example Execution
if __name__ == "__main__":

    with open('parameters.pkl', 'rb') as file:
        parameters = pickle.load(file)

    fixedLam = True
    if fixedLam:
        LAM = [ 6.93e1,  0.0 , 1.751e2 ,-1.192e1, 4.29e-3,  1.83e-1 , 2.159e-2 , 6.921e-1, 8.21e-6,  4.878e1 , 4.85e+1 , 1.5207e3]
        title = 'fixedLam'
    else:
        with open('ga_results.pkl', 'rb') as f:
            ga_data = pickle.load(f)
        LAM = ga_data['best_p_strings'][0]
        title = 'ga_result'

    sim = Simulation(parameters, LAM)
    hist_plane, hist_parts = sim.simulate_path_with_nozzles()
    print(f'Cost: {sim.calculate_cost():.4f}')
    # anim = Animation(hist_plane, hist_parts, parameters)
    # anim.animate_firefighting(title=title, savefig=True)

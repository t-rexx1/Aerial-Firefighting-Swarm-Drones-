"""
ME 144/244: Modeling, Simulation, and Digital Twins of Drone-Based Systems
Project 7: Aerial Firefighting
Spring 2026 Semester

Sensitivity analysis and plots.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

from simulation import Simulation


class AerialSensitivityAnalyzer:
    def __init__(self, parameters, animations_dir="../animations"):
        self.parameters = parameters
        self.animations_dir = animations_dir

    @staticmethod
    def from_parameters_file(parameters_path="parameters.pkl", animations_dir="../animations"):
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        return AerialSensitivityAnalyzer(parameters, animations_dir=animations_dir)

    @staticmethod
    def default_base_lam():
        return  [ 6.93e1,  0.0 , 1.751e2 ,-1.192e1, 4.29e-3,  1.83e-1 , 2.159e-2 , 6.921e-1, 8.21e-6,  4.878e1 , 4.85e+1 , 1.5207e3]

    @staticmethod
    def load_ga_best_lam(ga_results_path="ga_results.pkl", fallback=None):
        if fallback is None:
            fallback = AerialSensitivityAnalyzer.default_base_lam()
        try:
            with open(ga_results_path, "rb") as f:
                ga = pickle.load(f)
            best = ga["best_p_strings"][0]
            return list(np.array(best, dtype=float))
        except Exception:
            return list(fallback)

    def _run_grid(self, base_lam, sweep_index_x, x_values, sweep_index_y, y_values):
        x_values = np.array(x_values, dtype=float)
        y_values = np.array(y_values, dtype=float)

        X_grid, Y_grid = np.meshgrid(x_values, y_values)
        cost_grid = np.zeros_like(X_grid, dtype=float)

        total = X_grid.size
        print(f"Running {total} simulations...")

        for idx, (i, j) in enumerate(np.ndindex(X_grid.shape)):
            lam = list(np.array(base_lam, dtype=float))
            lam[sweep_index_x] = float(X_grid[i, j])
            lam[sweep_index_y] = float(Y_grid[i, j])

            sim = Simulation(self.parameters, lam)
            sim.simulate_path_with_nozzles()
            cost_grid[i, j] = sim.calculate_cost()

            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"  {idx + 1}/{total} done")

        print("Sweep complete.")
        return X_grid, Y_grid, cost_grid

    def _plot_surface_and_heatmap(
        self,
        X_grid,
        Y_grid,
        cost_grid,
        x_label,
        y_label,
        title,
        surface_filename,
        heatmap_filename,
    ):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=30, azim=60)
        surf = ax.plot_surface(
            X_grid,
            Y_grid,
            cost_grid,
            cmap="viridis",
            edgecolor="none",
            alpha=0.9,
        )

        ax.set_xlabel(x_label, labelpad=10)
        ax.set_ylabel(y_label, labelpad=10)
        ax.set_zlabel("Cost", labelpad=10)
        ax.set_title(title)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Cost")
        plt.tight_layout()

        surface_path = f"{self.animations_dir}/{surface_filename}"
        plt.savefig(surface_path, dpi=150)
        plt.show()
        print(f"Saved to {surface_path}")

        fig, ax = plt.subplots(figsize=(10, 7))
        heatmap = ax.contourf(X_grid, Y_grid, cost_grid, levels=40, cmap="viridis")
        ax.set_xlabel(x_label, labelpad=10)
        ax.set_ylabel(y_label, labelpad=10)
        ax.set_title(f"{title} (Heatmap)")
        plt.colorbar(heatmap, ax=ax, label="Cost")
        plt.tight_layout()

        heatmap_path = f"{self.animations_dir}/{heatmap_filename}"
        plt.savefig(heatmap_path, dpi=150)
        plt.show()
        print(f"Saved to {heatmap_path}")

    def run_velocity_drop_velocity(
        self,
        base_lam,
        velocity_values=None,
        drop_velocity_values=None,
        make_plots=True,
    ):
        if velocity_values is None:
            velocity_values = np.linspace(100 * 1000 / 3600, 300 * 1000 / 3600, 15)

        if drop_velocity_values is None:
            drop_velocity_values = np.linspace(0.0, 150.0, 15)

        V_grid, D_grid, cost_grid = self._run_grid(
            base_lam=base_lam,
            sweep_index_x=0,
            x_values=velocity_values,
            sweep_index_y=9,
            y_values=drop_velocity_values,
        )

        if make_plots:
            self._plot_surface_and_heatmap(
                V_grid,
                D_grid,
                cost_grid,
                x_label="Plane Velocity (m/s)",
                y_label="Particle Drop Velocity (m/s)",
                title="Sensitivity Analysis: Cost vs. Plane Velocity & Particle Drop Velocity",
                surface_filename="sensitivity_surface.png",
                heatmap_filename="sensitivity_heatmap.png",
            )

        return {
            "x_grid": V_grid,
            "y_grid": D_grid,
            "cost_grid": cost_grid,
            "x_values": np.array(velocity_values, dtype=float),
            "y_values": np.array(drop_velocity_values, dtype=float),
        }

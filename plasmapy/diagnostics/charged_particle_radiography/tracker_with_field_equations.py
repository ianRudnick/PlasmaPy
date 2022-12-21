"""
Child class of the Tracker class.
Allows the electric and magnetic fields to be defined by lambda functions,
instead of a grid of vectors, in order to remove the need for interpolation
and reduce the runtime of the simulation.
"""

__all__ = ["Tracker_With_Field_Funcs"]

from plasmapy.diagnostics.charged_particle_radiography import Tracker

# Copied from parent class in case we need any of this
import astropy.constants as const
import astropy.units as u
import numpy as np
import sys
import warnings

from tqdm import tqdm

from plasmapy import particles
from plasmapy.formulary.mathematics import rot_a_to_b
from plasmapy.particles import Particle
from plasmapy.plasma.grids import AbstractGrid
from plasmapy.simulation.particle_integrators import boris_push

class Tracker_With_Field_Funcs(Tracker):
    def __init__(
        self,
        grid: AbstractGrid,
        Ex_func,
        Ey_func,
        Ez_func,
        Bx_func,
        By_func,
        Bz_func,
        source: u.m,
        detector: u.m,
        detector_hdir=None,
        verbose=True,
    ):
        # These are used to avoid having to interpolate the fields for every
        # particle, which takes >90% of the execution time
        self.Ex_func = Ex_func        
        self.Ey_func = Ey_func
        self.Ez_func = Ez_func
        self.Bx_func = Bx_func
        self.By_func = By_func
        self.Bz_func = Bz_func

        # self.grid is the grid object
        self.grid = grid

        # self.grid_arr is the grid positions in si units. This is created here
        # so that it isn't continuously called later
        self.grid_arr = grid.grid.to(u.m).value

        self.verbose = verbose

        # A list of wire meshes added to the grid with add_wire_mesh
        # Particles that would hit these meshes will be removed at runtime
        # by _apply_wire_mesh
        self.mesh_list = []

        # This flag records whether the simulation has been run
        self._has_run = False

        # ************************************************************************
        # Setup the source and detector geometries
        # ************************************************************************

        self.source = _coerce_to_cartesian_si(source)
        self.detector = _coerce_to_cartesian_si(detector)
        self._log(f"Source: {self.source} m")
        self._log(f"Detector: {self.detector} m")

        # Calculate normal vectors (facing towards the grid origin) for both
        # the source and detector planes
        self.src_n = -self.source / np.linalg.norm(self.source)
        self.det_n = -self.detector / np.linalg.norm(self.detector)

        # Vector directly from source to detector
        self.src_det = self.detector - self.source

        # Magnification
        self.mag = 1 + (np.linalg.norm(self.detector) / np.linalg.norm(self.source))
        self._log(f"Magnification: {self.mag}")

        # Check that source-detector vector actually passes through the grid
        if not self.grid.vector_intersects(self.source * u.m, self.detector * u.m):
            raise ValueError(
                "The vector between the source and the detector "
                "does not intersect the grid provided!"
            )

        # Determine the angle above which particles will not hit the grid
        # these particles can be ignored until the end of the simulation,
        # then immediately advanced to the detector grid with their original
        # velocities
        self.max_theta_hit_grid = self._max_theta_hit_grid()

        # *********************************************************************
        # Define the detector plane
        # *********************************************************************

        # Load or calculate the detector hdir
        if detector_hdir is not None:
            self.det_hdir = detector_hdir / np.linalg.norm(detector_hdir)
        else:
            self.det_hdir = self._default_detector_hdir()

        # Calculate the detector vdir
        ny = np.cross(self.det_hdir, self.det_n)
        self.det_vdir = -ny / np.linalg.norm(ny)

        # *********************************************************************
        # Validate the E and B fields
        # *********************************************************************

        req_quantities = ["E_x", "E_y", "E_z", "B_x", "B_y", "B_z"]

        self.grid.require_quantities(req_quantities, replace_with_zeros=True)

        for rq in req_quantities:

            # Check that there are no infinite values
            if not np.isfinite(self.grid[rq].value).all():
                raise ValueError(
                    f"Input arrays must be finite: {rq} contains "
                    "either NaN or infinite values."
                )

            # Check that the max values on the edges of the arrays are
            # small relative to the maximum values on that grid
            #
            # Array must be dimensionless to re-assemble it into an array
            # of max values like this
            arr = np.abs(self.grid[rq]).value
            edge_max = np.max(
                np.array(
                    [
                        np.max(arr[0, :, :]),
                        np.max(arr[-1, :, :]),
                        np.max(arr[:, 0, :]),
                        np.max(arr[:, -1, :]),
                        np.max(arr[:, :, 0]),
                        np.max(arr[:, :, -1]),
                    ]
                )
            )

            if edge_max > 1e-3 * np.max(arr):
                unit = grid.recognized_quantities[rq].unit
                warnings.warn(
                    "Fields should go to zero at edges of grid to avoid "
                    "non-physical effects, but a value of "
                    f"{edge_max:.2E} {unit} was "
                    f"found on the edge of the {rq} array."
                    "Consider applying a envelope function to force the "
                    "fields at the edge to go to "
                    "zero.",
                    RuntimeWarning,
                )
    
    def _push(self):
        r"""
        Advance particles using an implementation of the time-centered
        Boris algorithm
        """
        # Get a list of positions (input for interpolator)
        pos = self.x[self.grid_ind, :] * u.m

        # Update the list of particles on and off the grid
        self.on_grid = self.grid.on_grid(pos)
        # entered_grid is zero at the end if a particle has never
        # entered the grid
        self.entered_grid += self.on_grid

        # Estimate the E and B fields for each particle
        # Note that this interpolation step is BY FAR the slowest part of the push
        # loop. Any speed improvements will have to come from here.
        # if self.field_weighting == "volume averaged":
        #     Ex, Ey, Ez, Bx, By, Bz = self.grid.volume_averaged_interpolator(
        #         pos,
        #         "E_x",
        #         "E_y",
        #         "E_z",
        #         "B_x",
        #         "B_y",
        #         "B_z",
        #         persistent=True,
        #     )
        # elif self.field_weighting == "nearest neighbor":
        #     Ex, Ey, Ez, Bx, By, Bz = self.grid.nearest_neighbor_interpolator(
        #         pos,
        #         "E_x",
        #         "E_y",
        #         "E_z",
        #         "B_x",
        #         "B_y",
        #         "B_z",
        #         persistent=True,
        #     )
        Ex = Ex_func(pos)
        Ey = Ey_func(pos)
        Ez = Ez_func(pos)
        Bx = Bx_func(pos)
        By = By_func(pos)
        Bz = Bz_func(pos)


        # Interpret any NaN values (points off the grid) as zero
        Ex = np.nan_to_num(Ex, nan=0.0 * u.V / u.m)
        Ey = np.nan_to_num(Ey, nan=0.0 * u.V / u.m)
        Ez = np.nan_to_num(Ez, nan=0.0 * u.V / u.m)
        Bx = np.nan_to_num(Bx, nan=0.0 * u.T)
        By = np.nan_to_num(By, nan=0.0 * u.T)
        Bz = np.nan_to_num(Bz, nan=0.0 * u.T)

        # Create arrays of E and B as required by push algorithm
        E = np.array(
            [Ex.to(u.V / u.m).value, Ey.to(u.V / u.m).value, Ez.to(u.V / u.m).value]
        )
        E = np.moveaxis(E, 0, -1)
        B = np.array([Bx.to(u.T).value, By.to(u.T).value, Bz.to(u.T).value])
        B = np.moveaxis(B, 0, -1)

        # Calculate the adaptive timestep from the fields currently experienced
        # by the particles
        # If user sets dt explicitly, that's handled in _adaptive_dt
        dt = self._adaptive_dt(Ex, Ey, Ez, Bx, By, Bz)

        # TODO: Test v/c and implement relativistic Boris push when required
        # vc = np.max(v)/_c

        x = self.x[self.grid_ind, :]
        v = self.v[self.grid_ind, :]
        boris_push(x, v, B, E, self.q, self.m, dt)
        self.x[self.grid_ind, :] = x
        self.v[self.grid_ind, :] = v
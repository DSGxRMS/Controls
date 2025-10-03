import pandas as pd
import numpy as np
from . import control_utils as utils

class PathManager:
    """
    Manages loading, preprocessing, and access to the race path.
    """
    def __init__(self, csv_path, scaling_factor, loop, path_offset_x, path_offset_y, num_arc_points=800):
        """
        Initializes the PathManager, loading and processing the path.
        """
        # Load the raw path from CSV
        path = pd.read_csv(csv_path)
        x_raw = path['x'].values
        y_raw = path['y'].values

        # Resample the track to uniform arc length
        xs, ys = utils.resample_track(x_raw, y_raw, num_arc_points)

        # Apply scaling and offsets
        xs = xs * scaling_factor + path_offset_x
        ys = ys * scaling_factor + path_offset_y
        
        # Pre-compute path data
        self.xs, self.ys, self.s, self.total_len = utils.preprocess_path(xs, ys, loop)
        
        # Compute and store additional path properties
        self.kappa_signed, self.seg_ds = utils.compute_signed_curvature(self.xs, self.ys)
        
        # Example of a global velocity limit based on curvature
        self.v_limit_global = np.ones_like(self.xs) * 2.0  # Default speed
        # Reduce speed in sharp turns
        self.v_limit_global[np.abs(self.kappa_signed) > 0.2] = 1.0

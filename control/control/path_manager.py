import pandas as pd
import numpy as np
from .controls_functions import resample_track, preprocess_path

class PathManager:
    """
    Manages loading, preprocessing, and storing of the path data.
    """
    def __init__(self, csv_path, scaling_factor, loop, path_offset_x, path_offset_y):
        """
        Initializes the PathManager, loads the path, and preprocesses it.

        Args:
            csv_path (str): Path to the CSV file containing the path data.
            scaling_factor (float): Scaling factor to apply to the path coordinates.
            loop (bool): Whether the path is a loop.
            path_offset_x (float): Offset to apply to the x-coordinates of the path.
            path_offset_y (float): Offset to apply to the y-coordinates of the path.
        """
        self.csv_path = csv_path
        self.scaling_factor = scaling_factor
        self.loop = loop
        self.path_offset_x = path_offset_x
        self.path_offset_y = path_offset_y

        self.xs = None
        self.ys = None
        self.s = None
        self.total_len = None
        self.kappa_signed = None

        self._load_and_preprocess_path()

    def _load_and_preprocess_path(self):
        """
        Loads the path from the CSV file, applies scaling and offsets,
        and performs resampling and preprocessing.
        """
        df = pd.read_csv(self.csv_path)
        
        # Apply scaling
        x_scaled = df["x"].to_numpy() * self.scaling_factor
        y_scaled = df["y"].to_numpy() * self.scaling_factor
        
        # Resample track
        rx, ry = resample_track(x_scaled, y_scaled)
        
        # Apply coordinate transformation and offsets
        route_x = rx + self.path_offset_x
        route_y = ry + self.path_offset_y
        
        self.xs, self.ys, self.s, self.total_len = preprocess_path(route_x, route_y, self.loop)
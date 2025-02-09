# base_augmenter.py
import numpy as np
import pandas as pd

class BaseAugmenter:
    def __init__(self, data_path):
        """Initialize with the VR dataset path"""
        self.data = pd.read_csv(data_path)
        self.numerical_columns = self._get_numerical_columns()
        self.sequence_length = 32  # Adjustable sequence length for time series

    def _get_numerical_columns(self):
        """Get numerical columns from the dataset"""
        return self.data.select_dtypes(include=[np.number]).columns

    def save_augmented_data(self, augmented_data, filename):
        """Save augmented data to CSV"""
        try:
            augmented_data.to_csv(filename, index=False)
            print(f"Successfully saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            return False
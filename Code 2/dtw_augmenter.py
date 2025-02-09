# dtw_augmenter.py
from base_augmenter import BaseAugmenter
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

class DTWAugmenter(BaseAugmenter):
    def _interpolate_sequences(self, seq1, seq2, path):
        """Interpolate between two sequences using DTW path"""
        interpolated = []
        for i, j in path:
            alpha = np.random.random()
            new_point = alpha * seq1.iloc[i] + (1 - alpha) * seq2.iloc[j]
            interpolated.append(new_point)
        return pd.DataFrame(interpolated)

    def augment(self, n_samples=100, output_file='dtw_augmented_data.csv'):
        """Dynamic Time Warping augmentation"""
        try:
            print("Starting DTW augmentation...")
            augmented_data = []
            
            for _ in range(n_samples):
                seq1_idx = np.random.randint(0, len(self.data) - self.sequence_length)
                seq2_idx = np.random.randint(0, len(self.data) - self.sequence_length)
                
                seq1 = self.data.iloc[seq1_idx:seq1_idx + self.sequence_length]
                seq2 = self.data.iloc[seq2_idx:seq2_idx + self.sequence_length]
                
                distance, path = fastdtw(seq1[self.numerical_columns], 
                                       seq2[self.numerical_columns], 
                                       dist=euclidean)
                
                new_sequence = self._interpolate_sequences(seq1, seq2, path)
                augmented_data.append(new_sequence)
            
            result = pd.concat(augmented_data)
            self.save_augmented_data(result, output_file)
            return result
            
        except Exception as e:
            print(f"DTW augmentation failed: {str(e)}")
            return None

if __name__ == "__main__":
    augmenter = DTWAugmenter('test2cleaned.csv')
    augmented = augmenter.augment(n_samples=100)
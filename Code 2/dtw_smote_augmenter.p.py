# dtw_smote_augmenter.py
from base_augmenter import BaseAugmenter
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

class DTWSMOTEAugmenter(BaseAugmenter):
    def _find_dtw_neighbors(self, base_seq, k):
        """Find k nearest neighbors using DTW distance"""
        distances = []
        for i in range(len(self.data) - self.sequence_length):
            curr_seq = self.data.iloc[i:i + self.sequence_length]
            distance, _ = fastdtw(base_seq[self.numerical_columns], 
                                curr_seq[self.numerical_columns], 
                                dist=euclidean)
            distances.append((distance, curr_seq))
        
        distances.sort(key=lambda x: x[0])
        return [seq for _, seq in distances[1:k+1]]

    def augment(self, n_samples=100, k_neighbors=5, output_file='dtw_smote_augmented_data.csv'):
        try:
            print("Starting DTW-SMOTE augmentation...")
            augmented_data = []
            
            for _ in range(n_samples):
                base_idx = np.random.randint(0, len(self.data) - self.sequence_length)
                base_seq = self.data.iloc[base_idx:base_idx + self.sequence_length]
                
                neighbors = self._find_dtw_neighbors(base_seq, k_neighbors)
                neighbor = neighbors[np.random.randint(0, len(neighbors))]
                
                alpha = np.random.random()
                new_sequence = alpha * base_seq + (1 - alpha) * neighbor
                augmented_data.append(new_sequence)
            
            result = pd.concat(augmented_data)
            self.save_augmented_data(result, output_file)
            return result
            
        except Exception as e:
            print(f"DTW-SMOTE augmentation failed: {str(e)}")
            return None

if __name__ == "__main__":
    augmenter = DTWSMOTEAugmenter('test2cleaned.csv')
    augmented = augmenter.augment(n_samples=100)
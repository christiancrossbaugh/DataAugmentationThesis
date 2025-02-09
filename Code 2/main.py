# main.py
from dtw_augmenter import DTWAugmenter
from cgan_augmenter import CGANAugmenter
from dtw_smote_augmenter import DTWSMOTEAugmenter
from llm_augmenter import LLMAugmenter
import pandas as pd

def main():
    # Initialize augmenters
    augmenters = {
        'DTW': DTWAugmenter('test2cleaned.csv'),
        'CGAN': CGANAugmenter('test2cleaned.csv'),
        'DTW-SMOTE': DTWSMOTEAugmenter('test2cleaned.csv'),
        'LLM': LLMAugmenter('test2cleaned.csv')
    }

    results = {}

    # Run each augmentation method
    for name, augmenter in augmenters.items():
        print(f"\nRunning {name} augmentation...")
        result = augmenter.augment(n_samples=100)
        if result is not None:
            results[name] = result

    # Create combined dataset
    if results:
        print("\nCreating combined dataset...")
        combined = pd.concat(results.values())
        combined.to_csv('all_augmented_data.csv', index=False)
        print("Combined dataset saved successfully.")

if __name__ == "__main__":
    main()
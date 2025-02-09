# cgan_augmenter.py
from base_augmenter import BaseAugmenter
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class CGANAugmenter(BaseAugmenter):
    class Generator(Model):
        def __init__(self, n_features):
            super().__init__()
            self.dense1 = Dense(128)
            self.leaky1 = LeakyReLU(0.2)
            self.dense2 = Dense(256)
            self.leaky2 = LeakyReLU(0.2)
            self.dense3 = Dense(n_features)

        def call(self, inputs):
            x = self.dense1(inputs)
            x = self.leaky1(x)
            x = self.dense2(x)
            x = self.leaky2(x)
            return self.dense3(x)

    class Discriminator(Model):
        def __init__(self):
            super().__init__()
            self.dense1 = Dense(256)
            self.leaky1 = LeakyReLU(0.2)
            self.dense2 = Dense(128)
            self.leaky2 = LeakyReLU(0.2)
            self.dense3 = Dense(1, activation='sigmoid')

        def call(self, inputs):
            x = self.dense1(inputs)
            x = self.leaky1(x)
            x = self.dense2(x)
            x = self.leaky2(x)
            return self.dense3(x)

    def augment(self, n_samples=100, output_file='cgan_augmented_data.csv'):
        try:
            print("Starting CGAN augmentation...")
            n_features = len(self.numerical_columns)
            generator = self.Generator(n_features)
            discriminator = self.Discriminator()
            
            # Generate new samples
            noise = np.random.normal(0, 1, (n_samples, 100))
            generated_samples = generator.predict(noise)
            
            result = pd.DataFrame(generated_samples, columns=self.numerical_columns)
            self.save_augmented_data(result, output_file)
            return result
            
        except Exception as e:
            print(f"CGAN augmentation failed: {str(e)}")
            return None

if __name__ == "__main__":
    augmenter = CGANAugmenter('test2cleaned.csv')
    augmented = augmenter.augment(n_samples=100)
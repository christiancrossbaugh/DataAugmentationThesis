# llm_augmenter.py
from base_augmenter import BaseAugmenter
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class LLMAugmenter(BaseAugmenter):
    def _convert_to_text(self):
        """Convert numerical data to text format"""
        return str(self.data[self.numerical_columns].values.tolist())

    def _convert_to_numerical(self, text):
        """Convert text back to numerical format"""
        try:
            values = eval(text)
            return pd.DataFrame(values, columns=self.numerical_columns)
        except:
            return None

    def augment(self, n_samples=100, output_file='llm_augmented_data.csv'):
        try:
            print("Starting LLM augmentation...")
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2LMHeadModel.from_pretrained('gpt2')
            
            augmented_data = []
            text_data = self._convert_to_text()
            
            for _ in range(n_samples):
                inputs = tokenizer.encode(text_data[:100], return_tensors='pt')
                outputs = model.generate(
                    inputs,
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.7
                )
                
                generated_sequence = self._convert_to_numerical(
                    tokenizer.decode(outputs[0])
                )
                if generated_sequence is not None:
                    augmented_data.append(generated_sequence)
            
            if augmented_data:
                result = pd.concat(augmented_data)
                self.save_augmented_data(result, output_file)
                return result
            return None
            
        except Exception as e:
            print(f"LLM augmentation failed: {str(e)}")
            return None

if __name__ == "__main__":
    augmenter = LLMAugmenter('test2cleaned.csv')
    augmented = augmenter.augment(n_samples=100)
# Sentiment Analysis on IMDB Movie Reviews

This repository contains a sentiment analysis project implemented using PyTorch. The goal of this project is to analyze customer movie reviews from the IMDB dataset and classify them as positive or negative.

## Dataset
The dataset used in this project is the IMDB movie reviews dataset. This dataset consists of labeled movie reviews that are used for training and evaluating the sentiment analysis model.

## Project Implementation Steps
1. **Tokenization**: Used `torchtext.get_tokenizer` for tokenizing text data.
2. **Vocabulary Creation**: Created a vocabulary storing words that appeared more than 5 times in the dataset.
3. **Dataset Splitting**: Split the dataset into training, testing, and validation subsets.
4. **Numerical Representation**: Converted bag-of-words representation into numerical indices.
5. **Tensor Conversion**: Transformed numerical data into PyTorch tensors.
6. **Word Embeddings**: Converted tensors into word embeddings with 300 dimensions.
7. **Neural Bag of Words Model**: Used a neural bag of words neural network for prediction.
8. **Pre-trained Word Embeddings**: Used pre-trained word embeddings present in `torchtext` called GloVe.
9. **Model Training**: Used `nn.Linear` model for training and predicting.
10. **Batch Processing**: Used batch sizing to convert the sentences into a tensor of sentences with batch size 512.
11. **Pooling**: Applied pooling for each batch to get the final representation.
12. **Loss Function & Activation**: Used `CrossEntropyLoss` as the criterion, along with `argmax`, `softmax` for final output and predicted probabilities.

## Dependencies
To run this project, install the required dependencies:

```bash
pip install torch torchtext datasets tqdm numpy matplotlib
```

## Required Libraries
```python
import collections
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import datasets
import tqdm
```

## Running the Project
Execute the Jupyter Notebook (`.pynb` file) in a Python environment with the necessary libraries installed.

## Model Training
The model is trained using PyTorch with an embedding layer of 300 dimensions and optimized using appropriate loss functions and optimizers.

## Results
The model effectively classifies movie reviews as positive or negative based on the provided training data.

## Contributions
Feel free to contribute by improving the model, adding new features, or optimizing the implementation.

## License
This project is open-source and available under the [MIT License](LICENSE).

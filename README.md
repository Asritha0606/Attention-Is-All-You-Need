
# Attention Is All You Need – Transformer Model Implementation

## Overview

This repository provides a detailed implementation of the Transformer model as introduced in the seminal paper **"Attention Is All You Need"** by Vaswani et al. The Transformer architecture uses self-attention mechanisms to excel at sequence-to-sequence tasks such as machine translation, bypassing traditional recurrent or convolutional layers.

Alongside the model implementation, this repository includes:

- Code to download and preprocess an English-French translation dataset from Kaggle
- Training pipeline with batching, masking, and evaluation utilities
- An inference example that demonstrates how to generate translations from input text

## The Paper: Attention Is All You Need

The paper introduces a new architecture for machine translation based solely on attention mechanisms:

- **Self-Attention:** Enables the model to weigh relationships between words in a sequence regardless of their positional distance.
- **Multi-Head Attention:** Allows the model to jointly attend to information from different representation subspaces.
- **Positional Encoding:** Injects sequence order information into the input embeddings.
- **Encoder & Decoder Stacks:** Both constructed from layers consisting of multi-head attention and position-wise feedforward networks.
- **Advantages:** Parallelizable training, superior performance on benchmark tasks, and better capability to capture long-range dependencies.

For deeper understanding, please refer to the original paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

## Repository Structure

- **Transformer Implementation**: Contains classes for Scaled Dot-Product Attention, Multi-Head Attention, Encoder and Decoder layers, Positional Encoding, and the overall Transformer model.
- **Dataset Handling**: Uses `kagglehub` to download the English-French translation dataset, with custom PyTorch Dataset and DataLoader utilities to preprocess and batch data.
- **Training Script**: Implements a training loop with cross-entropy loss and Adam optimizer.
- **Inference Example**: Shows how to generate translated text from an English input sentence using the trained model and BERT tokenizer.

## How to Use

### 1. Setup Environment

Install required packages:
pip install torch transformers pandas kagglehub


### 2. Download Dataset

The dataset is automatically downloaded via `kagglehub` in the training script.

### 3. Run Training

Run the training script. It will:
- Load and preprocess the dataset (English-French pairs)
- Train the Transformer model for a configurable number of epochs
- Print training loss per epoch

### 4. Perform Inference

After training, the script includes an inference example that:
- Takes an English sentence as input
- Tokenizes and passes it through the Transformer
- Generates a translated sentence in French

### 5. Modify Dataset Size & Hyperparameters

For better results, increase the dataset size by adjusting:
```python
data = data[['English words/sentences', 'French words/sentences']].dropna().head(10000)  # or more

and tune hyperparameters such as number of epochs, learning rate, and batch size.


###Example Model Usage


test_sentence = "hello"


translated_text = model.translate(test_sentence)


print(f"Translated text: {translated_text}")

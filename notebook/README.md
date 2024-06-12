## Transformer Model Implementation using PyTorch

This [Notebook](./Transformer_In_PyTorch.ipynb) contains the experimentation of a Transformer model using PyTorch.

A Transformer model typically consists of an encoder and a decoder, both of which are built using multi-head attention mechanisms and feedforward neural networks.

### Components

1. **Positional Encoding**

   - Adds positional information to the input embeddings to retain the order of the sequence since the model is inherently order-agnostic.

2. **Multi-Head Attention**

   - Allows the model to focus on different parts of the input sequence simultaneously, enhancing its ability to capture relationships between words.

3. **Feed Forward Network**

   - A two-layer fully connected neural network applied to each position separately and identically.

4. **Encoder Layer**

   - Comprises a multi-head attention mechanism and a feedforward network, each followed by layer normalization and residual connections.

5. **Decoder Layer**

   - Similar to the encoder layer but includes an additional multi-head attention mechanism to attend to the encoder's output.

6. **Encoder**

   - Stacks multiple encoder layers. Encodes the input sequence into a continuous representation.

7. **Decoder**

   - Stacks multiple decoder layers. Decodes the encoded representations into the output sequence.

8. **Transformer Model**
   - Combines the encoder and decoder. Includes masking mechanisms to prevent the model from attending to future tokens during training.

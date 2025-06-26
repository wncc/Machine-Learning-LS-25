# Transformers
Transformers are a type of deep learning model introduced in the 2017 paper ["Attention is All You Need" by Vaswani et al](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf) . They revolutionized natural language processing (NLP) and are now used in many domains like vision, speech, and more.

Applications:
- NLP: BERT, GPT, T5, etc.

- Vision: Vision Transformers (ViT)

- Speech, Recommendation Systems, Genomics, etc.

## Recurrent Neural network (RNN):
In order to understand the Transformer Architecture, It is Important to understand the RNN Stucture

[Introduction to RNN](https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/)

[More Indepth Intution](https://www.geeksforgeeks.org/machine-learning/recurrent-neural-networks-explanation/)

[Video Resource 1](https://www.youtube.com/watch?v=Y2wfIKQyd1I)

[Video Resource 2](https://www.youtube.com/watch?v=EzsXi4WzelI)

[Bi-directional RNN](https://www.geeksforgeeks.org/bidirectional-recurrent-neural-network/)

[Video Resource](https://youtu.be/atYPhweJ7ao?si=VlHMh2zwwqPWo5AF)

## LSTM RNN
Traditional RNNs struggle with vanishing gradients, making them bad at remembering information over long sequences.
LSTM solves this using a memory cell and gates that control information flow.

[Introduction to LSTM](https://www.geeksforgeeks.org/machine-learning/long-short-term-memory-networks-explanation/)

[More Indepth Intution](https://www.geeksforgeeks.org/machine-learning/long-short-term-memory-networks-explanation/)

[Video Resources](https://youtu.be/LfnrRPFhkuY?si=Q7tAaOkHZUwokVAV)

[GRU-RNN (optional)](https://youtu.be/tOuXgORsXJ4?si=Cv2JxQQip_lmi40h)

## Encoder-Decoder
The Encoder-Decoder is a neural network architecture designed to handle sequence-to-sequence (seq2seq) tasks — where input and output are both sequences, possibly of different lengths.

[Encoder-Decoder (Seq2Seq)](https://medium.com/analytics-vidhya/encoder-decoder-seq2seq-models-clearly-explained-c34186fbf49b)

[Video Resources](https://www.youtube.com/watch?v=L8HKweZIOmg)

## Attention Mechanism
The attention mechanism allows a model to focus on relevant parts of the input sequence when generating each part of the output.

It was introduced to improve performance in sequence-to-sequence tasks (like translation), especially in long sequences.

[Attention Mechanism Explained](https://erdem.pl/2021/05/introduction-to-attention-mechanism)

[Video Resources](https://youtu.be/PSs6nxngL6k?si=mcrMj7QjLftvYXz4)


## Transformers
### Key Concepts:
- [Self-Attention](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html) :
  Allows the model to weigh the importance of different words in a sentence when encoding a specific word.

- [Positional Encoding](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/) :
Since transformers don't process sequences in order (like RNNs), positional encodings are added to retain word order.

### Core Components:
- Multi-Head Attention:
Captures relationships from different subspaces in the data.

- Feed-Forward Neural Networks:
Applies transformations to each position independently.

- Layer Normalization & Residual Connections:
Helps with training stability and deeper networks.

[Transformer Architecture](https://jalammar.github.io/illustrated-transformer/)

[Research Paper (Must Read)](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf)

[Video Resources](https://youtu.be/ZhAz268Hdpw?si=FCy2wMu-hOBIxIIu)

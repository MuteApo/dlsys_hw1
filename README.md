# Homework 1
Public repository and stub/testing code for Homework 1 of 10-714.

## Dev Environment
- Windows 11
- Miniconda3 (with Python 3.10.6)

## Hints

Question 2: Implementing backward computation
- Notice `.shape` consistency between gradients and their corresponding inputs.
- `Reshape()`, `BroadcastTo()` and `Summation()` are tightly coupled and may invoke each other in `.gradient()`.
- `Summation()` takes only `tuple` as input to `.axes`, supporting `int` (1-axis case) may be helpful.
- `Matmul()` should support matrix multiplication both batched or not.

Question 4: Implementing reverse mode differentiation
- Gradient is stored in `.grad` of one `Tensor`.
- Leaf nodes also need gradient computation.

Question 6: SGD for a two-layer neural network
- Invoke `.backward()` of loss function to start gradient computation.
- Weight updating does not extend computational graph (thus do not require gradient), stop gradient computation by invoking `.detach()` of the `Tensor`.
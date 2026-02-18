# Deep Learning from Scratch

A complete deep learning curriculum built from first principles. Every notebook is self-contained and designed to be run in Google Colab. The series starts with the math you need, builds neural networks by hand, then covers the three major frameworks used in industry and research today.

---

## Watch the Series


[Pytorch Zero to NN](https://youtu.be/h__d70F8E7w)
[Advance Tensorflow and Keras](https://youtu.be/uqYiYEIHNBM)
[Calculus](https://youtu.be/00h4fPcry8w)
[DL Geometry Matters](https://youtu.be/dsmIsxqFUCs)
[Jax Intro](https://youtu.be/-WrnHZKG1bE)
[Jax Indepth](https://youtu.be/Nm40Lq-n8ao)
[Neural Network](https://youtu.be/a9KUgYYCZLE)
[Numpy and Linear Algebra](https://youtu.be/UYaawG_LNus)
[Probability](https://youtu.be/HrkdiuSVWPc)
[Pytorch Advance](https://youtu.be/MEvBVuIvSiw)


---

## Course Structure

### Part 1: Mathematical Foundations

These four notebooks cover the math that underpins all of deep learning. If you are completely new, start here. If you have a math background, you can skim or skip ahead.

**NumPy Foundations**
Arrays, broadcasting, matrix multiplication, and vectorized operations. Every deep learning framework builds on these concepts, and NumPy is the clearest place to learn them.

**Linear Algebra**
Vectors, matrices, dot products, linear transformations, norms, and eigendecomposition. A neural network layer is a linear transformation followed by a nonlinearity — this notebook makes that concrete.

**Calculus**
Derivatives, partial derivatives, the chain rule, and backpropagation derived from first principles. This is the mathematical engine behind all gradient-based learning.

**Probability**
Random variables, probability distributions, Bayes' theorem, maximum likelihood estimation, entropy, and cross-entropy. Understanding why cross-entropy is the right loss function for classification requires this material.

---

### Part 2: Understanding Neural Networks

**Neural Networks from Scratch**
A complete neural network built using only NumPy — no frameworks. Forward propagation, activation functions, loss computation, backpropagation, and gradient descent are all implemented by hand. The goal is to leave you with no black boxes.

**Why Deep Learning Works: Geometric Intuition**
An explanation of deep learning through the lens of geometry. Each layer folds and warps the input space, progressively restructuring data until a simple decision boundary becomes possible. Includes visualizations of how depth provides expressiveness that width alone cannot match.

---

### Part 3: PyTorch

**PyTorch Tensors: From Zero to Hero**
A thorough introduction to PyTorch tensors — creation, shapes, dtypes, devices, indexing, reshaping, broadcasting, and matrix operations. Also introduces autograd: how PyTorch tracks operations to compute gradients automatically.

**PyTorch Neural Networks**
Building and training neural networks with `nn.Module`. Covers linear layers, activation functions, dropout, batch normalization, optimizers, loss functions, data loading with `DataLoader`, and saving and loading model checkpoints.

**PyTorch Advanced: Custom Layers and Autograd**
Higher-order derivatives, Jacobians, Hessians, and custom gradient functions with `torch.autograd.Function`. Building custom `nn.Module` subclasses, residual connections, attention mechanisms, custom loss functions, gradient clipping, and learning rate scheduling.

---

### Part 4: TensorFlow and Keras

**TensorFlow and Keras: Tensors to Neural Networks**
Covers the complete TensorFlow workflow in one track: tensor operations and the `tf.data` pipeline, the three Keras API styles (Sequential, Functional, and subclassing), GradientTape for manual gradient computation, `tf.function` for graph compilation, and production training patterns with callbacks.

---

### Part 5: JAX

**JAX for Deep Learning**
JAX combines a NumPy-compatible API with four composable transformations: `grad` for automatic differentiation, `jit` for XLA compilation, `vmap` for automatic vectorization, and `pmap` for multi-device parallelism. This track covers JAX fundamentals, einsum notation, building neural networks with Flax, and optimization with Optax.

---

## Prerequisites

- Python 3.8 or higher
- Basic familiarity with Python syntax
- No prior deep learning knowledge required

All notebooks run in Google Colab without any local setup. GPU runtimes are recommended for the framework notebooks and are available for free in Colab.

---

## Running the Notebooks

Each notebook can be opened directly in Colab via the badge at the top of the file, or by uploading it manually.

To run locally:

```bash
git clone https://github.com/your-username/deep-learning-from-scratch
cd deep-learning-from-scratch
pip install numpy matplotlib torch torchvision tensorflow keras jax flax optax
jupyter notebook
```

---

## Recommended Order

If you are new to deep learning, follow the numbered order above. If you have prior experience:

- Skip to Part 2 if you are comfortable with linear algebra, calculus, and probability
- Skip to Part 3, 4, or 5 if you already understand how neural networks work and want to learn a specific framework
- The framework tracks (PyTorch, TensorFlow, JAX) are independent of each other and can be done in any order

---

## Contents

```
.
├── numpy_foundations_for_deep_learning.ipynb
├── linear_algebra_for_deep_learning.ipynb
├── calculus_for_deep_learning.ipynb
├── probability_for_deep_learning.ipynb
├── probability_fundamentals_for_deep_learning.ipynb
├── Colab_1_neural_networks_from_scratch.ipynb
├── why_deep_learning_works_geometric_intuition.ipynb
├── colab_4_pytorch_tensors_from_zero_to_hero.ipynb
├── colab_2_pytorch_neural_networks_tutorial.ipynb
├── pytorch_advanced_tutorial.ipynb
├── tensorflow_tensor_operations_tutorial.ipynb
├── colab_3_keras_tensorflow_neural_networks_tutorial.ipynb
├── keras_tensorflow_advanced_tutorial.ipynb
├── colab_4_jax_neural_networks_tutorial.ipynb
└── colab_5_jax_deep_learning_tutorial.ipynb
```

---

## License

MIT

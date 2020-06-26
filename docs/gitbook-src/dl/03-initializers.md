# Initializer

An `Initializer` takes the shape of a tensor and returns a `float64`
numpy array of that shape. For example, 
`kernel_initialilzer=lambda s: np.ones(shape=s)` is a valid initializer.

### Types of initializer

+ **Kernel Initializer**: Specifies how to initialize the weights.
+ **Bias Initializer**: Specifies how to initialize the bias.

### Predefined Initializer

Here is a list of predefined initializers:

| Name | Description | Extra Argument(s) |
| ---- | ----------- | ----------------- |
| `zeros` | All 0s     | - |
| `ones`  | All 1s     | - |
| `const` | All `n`s   | `n: float = 1.0` |
| `random_normal` | Random normal | `mean: float = 0.0`, `stddev: float = 0.05` |
| `random_uniform` | Random uniform | `minval: float = -0.05`, `maxval: float = 0.05` |
| `glorot_normal` | `sqrt(2 / (prod(shape[:-1]) + shape[-1]))` | - |
| `glorot_uniform` | `sqrt(6 / (prod(shape[:-1]) + shape[-1]))` | - |

# Regularizers

A `regularizer` is a function of type `tensor -> tensor`.
The output is summed together with the loss function to form the terminal
node of the loss function.

For example, when `kernel_regularizer` `my_l1` is applied to a dense
layer, its weights `w` is then applied, forming `my_l1(w)`. 
It is then added to the loss function, 
forming `loss(prediction) + my_l1(w)`, which is the 
final "objective value".

### Types of regularizer

+ **Kernel Regularizer**: regularizing the weights (kernel).
+ **Bias Regularizer**: regularizing the bias.
+ **Activity Regularizer**: regularzing the layer's output (after applying activation function).

### Predefined regularizers

| Name | Parameters | Description |
| ---- | ---------- | ----------- |
| `l1` | `l1=0.01`(λ)  | f(x) = λ Σ&#124;x&#124; |
| `l2` | `l2=0.01` (λ) | f(x) = λ Σx<sup>2</sup> |
| `l1_l2` | `l1=0.01`(λ<sub>1</sub>), `l2=0.01`(λ<sub>2</sub>) | f(x) = λ<sub>1</sub> Σ&#124;x&#124; + λ<sub>2</sub> Σx<sup>2</sup> |

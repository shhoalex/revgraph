# Optimizers

An `optimizer` is a function that takes a tensor and return another 
tensor that's derived from the class `rc.optimizer`. Its interface
is exactly the same as the ones in 
[differentiable functions](../core/03-differentiable-functions.md).
Only this time the `minimize` target doesn't need to be specified.

### Predefined Optimizers

| Name | Arguments |
| ---- | ---------- | 
| `sgd` | `lr=0.001`, `momentum=0.0`, `decay=0.0`, `nesterov=False` |
| `rmsprop` | `lr=0.001`, `rho=0.9`, `epsilon=1e-9`, `decay=0.0` |
| `adagrad` | `lr=0.001`, `epsilon=1e-9`, `decay=0.0` |
| `adadelta` | `lr=1.0`, `rho=0.95`, `epsilon=1e-6`, `decay=0.0` | 
| `adam` | `lr=0.001`, `beta1=0.9`, `beta2=0.999`, `amsgrad=False`, `epsilon=1e-6`, `decay=0.0` |

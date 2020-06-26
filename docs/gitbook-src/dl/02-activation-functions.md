# Activation Functions

An `ActivationFunction` is a function of type `tensor -> tensor`. 

In the `activation` field of `dense` and `conv2d`, it accepts an `ActivationFunction`
and applies it to the original output. For example: `activation=lambda x: x/2` is valid
since it returns a tensor as output.

Here is a list of pre-defined activation functions (unfortunately mathjax is buggy when using gitbook):

| Name | Description | Extra Argument(s) |
| ---- | ----------- | ----------------- |
| `dl.linear` | f(x) = x | - |
| `dl.tanh` | f(x) = tanh(x) | - |
| `dl.sigmoid` | f(x) = 1 / (1 + exp(-x)) | - |
| `dl.relu` | f(x) = if x>0 then x else 0 | - |
| `dl.softmax` | f(x<sub>i</sub>) = exp(x<sub>i</sub>) / sum(exp(x)) | `axis`, `keepdims` |
| `dl.softplus` | f(x) = ln(exp(x) + 1) | - |

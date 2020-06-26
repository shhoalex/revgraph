# Metrics

A `metric` is a function that indicates how good/bad the model
is performing. **Any loss function can also be used as metric.** 
Metrics won't affect the optimization process of the model at all.

#### Predefined Metric

The metric `categorical_accuracy` is defined for evaluating the 
performance of multiclass classification.

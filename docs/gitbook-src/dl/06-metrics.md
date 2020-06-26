# Metrics

A `metric` is a function that provides metric for how good/bad the model
is performing. **Any loss function can also be used as metric.** 

The metric is simply an indicator of the performance of the model, 
it won't affect the optimization process of the model at all.

#### Predefined Metric

The metric `categorical_accuracy` is defined for evaluating the 
performance of multiclass classification.

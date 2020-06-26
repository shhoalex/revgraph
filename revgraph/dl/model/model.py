from contextlib import contextmanager

import dill as pkl

from revgraph.dl.core.utils import *

from .batch_generator import batch_generator
from .train_test_validation_split import train_test_validation_split


class Model(object):
    """
    Wrapper class for combining all the sub-components such as layers, loss
    function, metrics and optimizer.
    """
    def __init__(self,
                 model: GraphBuilderNoParam,
                 loss: GraphBuilder,
                 optimizer: GraphBuilder,
                 metrics: Dict[str, GraphBuilder] = None):
        self.model_builder = model
        self.loss_builder = loss
        self.optimizer_builder = optimizer
        self.metrics_builder = metrics if metrics is not None else {}
        self.compiled = False
        self.prediction = None
        self.loss = None
        self.metrics = None
        self.optimization = None

    @staticmethod
    def builder_not_found(*args, **kwargs):
        raise RuntimeError('Unable to compile a loaded model')

    @staticmethod
    def load_from(path: str) -> 'Model':
        with open(path, 'rb') as handler:
            nodes = pkl.load(handler)
            new_session = Model(
                model=Model.builder_not_found,
                loss=Model.builder_not_found,
                optimizer=Model.builder_not_found,
                metrics=Model.builder_not_found
            )
            new_session.prediction = nodes['prediction']
            new_session.loss = nodes['loss']
            new_session.optimization = nodes['optimization']
            new_session.metrics = nodes['metrics']
            new_session.compiled = True
            return new_session

    def compile(self):
        """
        Compiling the high level specifications to 1 big computational graph.
        """
        self.compiled = True
        metadata = self.model_builder()
        output_shape = ((metadata['units'],)
                        if isinstance(metadata['units'], int)
                        else metadata['units'])
        y_pred = self.prediction = get_graph(metadata)
        y_true = rc.placeholder(name='y', shape=(None, *output_shape))
        regularized_nodes = metadata['regularized_nodes']
        self.loss = self.loss_builder(y_true, y_pred)
        self.metrics = {k: f(y_true, y_pred) for k, f in self.metrics_builder.items()}
        if regularized_nodes is not None:
            self.loss += regularized_nodes
        self.optimization = self.optimizer_builder(self.loss)
        return self

    def predict(self, x):
        return self.prediction(x=x)

    def predict_one(self, x):
        return self.prediction(x=[x])[0]

    def get_model(self):
        return self.predict

    def evaluate(self, x, y):
        return self.loss(x=x, y=y)

    def evaluate_metrics(self, x, y):
        return {k: f(x=x, y=y) for k, f in self.metrics.items()}

    @staticmethod
    def invoke_callbacks(callbacks, metadata):
        for callback in callbacks:
            callback.send_metadata(**metadata)
            callback.invoke()

    @staticmethod
    @contextmanager
    def temporary(metadata, field, before=True, after=False):
        """
        Temporarily sets a field in metadata.
        """
        try:
            metadata[field] = before
            yield metadata
        finally:
            metadata[field] = after

    def fit(self,
            x: Union[np.ndarray, Iterator[Any]],
            y: Union[np.ndarray, Iterator[Any]],
            epochs: int = 1,
            batch_size: Optional[int] = None,
            shuffle: bool = True,
            train_test_validation: Tuple[float, float, float] = (1.0, 0.0, 0.0),
            verbose: bool = True,
            callbacks: List['base_callback'] = None):
        """
        The "train loop" of the model that integrates with callbacks.
        """
        callbacks = callbacks if callbacks is not None else []
        output = print if verbose else (lambda *args, **kwargs: None)
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        validate((
            self.compiled,
            'session is not compiled, try calling .compile'
        ), (
            len(x) == len(y),
            f'len(x) {len(x)} != len(y) {len(y)}'
        ))
        train, test, validation = train_test_validation
        (x_train, y_train), (x_test, y_test), (x_validation, y_validation) = (
            train_test_validation_split(x, y, train, test, validation)
        )
        batch_size = len(x_train) if batch_size is None else batch_size
        metadata = {
            'n_epochs': epochs,
            'n_batches': len(x_train) // batch_size,
            'session': self,
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test,
            'before_batch': False,
            'after_batch': False,
            'before_all': True,
            'after_all': False,
            'before_epoch': False,
            'after_epoch': False,
            'epoch': None,
            'batch': None,
            'x_validation': x_validation,
            'y_validation': y_validation,
            'output': output
        }

        with self.temporary(metadata, 'before_all') as m:
            self.invoke_callbacks(callbacks, m)

        for n in range(epochs):
            metadata['epoch'] = n
            with self.temporary(metadata, 'before_epoch') as m:
                self.invoke_callbacks(callbacks, m)
            for i, (x_batch, y_batch) in enumerate(batch_generator(x_train, y_train, batch_size, shuffle)):
                metadata['batch'] = i
                metadata['x_batch'] = x_batch
                metadata['y_batch'] = y_batch
                with self.temporary(metadata, 'before_batch') as m:
                    self.invoke_callbacks(callbacks, m)
                    self.optimization(x=x_batch, y=y_batch)
                with self.temporary(metadata, 'after_batch') as m:
                    self.invoke_callbacks(callbacks, m)
            with self.temporary(metadata, 'after_epoch') as m:
                self.invoke_callbacks(callbacks, m)

        with self.temporary(metadata, 'after_all') as m:
            self.invoke_callbacks(callbacks, m)

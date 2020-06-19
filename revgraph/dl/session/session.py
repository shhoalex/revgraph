from contextlib import contextmanager

from revgraph.dl.core.utils import *

from .batch_generator import batch_generator
from .train_test_validation_split import train_test_validation_split


class Session(object):
    def __init__(self,
                 model: GraphBuilderNoParam,
                 loss: GraphBuilder,
                 optimizer: GraphBuilder,
                 verbose: bool = False):
        self.model_builder = model
        self.loss_builder = loss
        self.optimizer_builder = optimizer
        self.compiled = False
        self.prediction = None
        self.verbose = verbose
        self.loss = None
        self.optimization = None

    def compile(self):
        self.compiled = True
        metadata = self.model_builder()
        output_shape = ((metadata['units'],)
                        if isinstance(metadata['units'], int)
                        else metadata['units'])
        self.prediction = get_graph(metadata)
        regularized_nodes = metadata['regularized_nodes']
        self.loss = self.loss_builder(
            rc.placeholder(name='y', shape=(None, *output_shape)),
            self.prediction
        )
        if regularized_nodes is not None:
            self.loss += regularized_nodes
        self.optimization = self.optimizer_builder(self.loss)

    def predict(self, x):
        return self.prediction(x=x)

    def get_model(self):
        return self.predict

    def evaluate(self, x, y):
        return self.loss(x=x, y=y)

    @staticmethod
    def invoke_callbacks(callbacks, metadata):
        for callback in callbacks:
            callback.send_metadata(**metadata)
            callback.invoke()

    @staticmethod
    @contextmanager
    def temporary(metadata, field, before=True, after=False):
        try:
            metadata[field] = before
            yield metadata
        finally:
            metadata[field] = after

    def fit(self,
            x: np.array,
            y: np.array,
            epochs: int = 1,
            batch_size: Optional[int] = None,
            shuffle: bool = True,
            train_test_validation: Tuple[float, float, float] = (1.0, 0.0, 0.0),
            verbose: bool = True,
            callbacks: List['base_callback'] = None):
        callbacks = callbacks if callbacks is not None else []
        output = print if verbose else (lambda *args, **kwargs: None)

        validate((
            self.compiled,
            'session is not compiled, try calling .compile'
        ))
        train, test, validation = train_test_validation
        (x_train, y_train), (x_test, y_test), (x_validation, y_validation) = (
            train_test_validation_split(x, y, train, test, validation)
        )
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

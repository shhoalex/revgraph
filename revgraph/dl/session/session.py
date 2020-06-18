from revgraph.dl.core.utils import *

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
        self.loss = self.loss_builder(
            rc.placeholder(name='y', shape=(None, *output_shape)),
            self.prediction
        )
        self.optimization = self.optimizer_builder(self.loss)

    def predict(self, x):
        return self.prediction(x=x)

    def get_model(self):
        return self.predict

    def evaluate(self, x, y):
        return self.loss(x=x, y=y)

    def fit(self,
            x: np.array,
            y: np.array,
            epochs: int = 1,
            batch_size: Optional[int] = None,
            shuffle: bool = True,
            train_test_validation: Tuple[float, float, float] = (1.0, 0.0, 0.0)):
        validate((
            self.compiled,
            'session is not compiled, try calling .compile'
        ))
        train, test, validation = train_test_validation
        (x_train, y_train), (x_test, y_test), (x_validation, y_validation) = (
            train_test_validation_split(x, y, train, test, validation)
        )

        for n_epoch in range(epochs):
            pass


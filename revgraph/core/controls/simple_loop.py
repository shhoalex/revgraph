from revgraph.core.base.computation import Computation


class SimpleLoop(Computation):
    def __init__(self, n, action, feed_dict=None):
        super().__init__()
        self.iterations = n
        self.action = action
        self.dependencies = action.dependencies
        self.data = None
        self.feed_dict = feed_dict

    def forward(self):
        feed_iter_to_dict = callable(self.feed_dict)
        for i in range(self.iterations):
            if self.feed_dict:
                if feed_iter_to_dict:
                    self.action(**self.feed_dict(i))
                else:
                    self.action(**self.feed_dict)
            else:
                self.action()

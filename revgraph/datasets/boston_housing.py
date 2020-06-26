import os

import numpy as np

# Source: http://lib.stat.cmu.edu/datasets/boston


def load_data(as_dict: bool = False,
              train_test_split: float = 0.75
              ):
    base_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_path, 'data/boston_housing/boston-housing.data')) as fs:
        lines = fs.readlines()
        data = np.array([[float(value)
                          for value in line[:-1].split(' ')
                          if value.strip()]
                         for line in lines])
        if as_dict:
            return [{
                'CRIM': d[0],
                'ZN': d[1],
                'INDUS': d[2],
                'CHAS': d[3],
                'NOX': d[4],
                'RM': d[5],
                'AGE': d[6],
                'DIS': d[7],
                'RAD': d[8],
                'TAX': d[9],
                'PTRATIO': d[10],
                'B': d[11],
                'LSTAT': d[12],
                'MEDV': d[13]
            } for d in data]

        x, y = data[:, :13], data[:, 13].reshape(-1)
        n_records = len(x)
        train_slice = slice(0, int(n_records * train_test_split), None)
        test_slice = slice(int(n_records * train_test_split), None, None)
        x_train, y_train = x[train_slice], y[train_slice]
        x_test, y_test = x[test_slice], y[test_slice]
        return (x_train, y_train), (x_test, y_test)


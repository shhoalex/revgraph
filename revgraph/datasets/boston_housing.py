import os

import numpy as np


def load_data(as_dict: bool = False):
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

        return data

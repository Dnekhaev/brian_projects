import numpy as np

digits = [[[0, 0, 1, 0, 0],
               [0, 1, 0, 1, 0],
               [0, 1, 0, 1, 0],
               [0, 1, 0, 1, 0],
               [0, 0, 1, 0, 0],
             ],
              [[0, 0, 1, 0, 0],
               [0, 1, 1, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0],
             ],
              [[0, 1, 1, 1, 0],
               [0, 0, 0, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 0, 0, 0],
               [0, 1, 1, 1, 0],
             ],
              [[0, 1, 1, 1, 0],
               [0, 0, 0, 1, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 1, 0],
               [0, 1, 1, 1, 0],
             ],
              [[0, 1, 0, 1, 0],
               [0, 1, 0, 1, 0],
               [0, 1, 1, 1, 1],
               [0, 0, 0, 1, 0],
               [0, 0, 0, 1, 0],
             ],
              [[0, 1, 1, 1, 0],
               [0, 1, 0, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 0, 1, 0],
               [0, 1, 1, 1, 0],
             ],
              [[0, 1, 1, 1, 0],
               [0, 1, 0, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 0, 1, 0],
               [0, 1, 1, 1, 0],
             ],
              [[0, 1, 1, 1, 0],
               [0, 0, 0, 1, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0],
             ],
              [[0, 1, 1, 1, 0],
               [0, 1, 0, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 0, 1, 0],
               [0, 1, 1, 1, 0],
             ],
              [[0, 1, 1, 1, 0],
               [0, 1, 0, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 0, 1, 0],
               [0, 1, 1, 1, 0],
             ]
             ]

def stupid_digits_dataset(num = 1000):
    X = []
    y = []
    for i in np.arange(num):
        index = np.random.randint(10)
        X.append(digits[index])
        y.append(index)
        
    X = np.array(X).reshape(num, 25)
    y = np.array(y)
    
    return X, y
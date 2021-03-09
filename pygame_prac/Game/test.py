import numpy as np

def connect(x1, y1, x2, y2):
    dx, dy = abs(x2-x1), abs(y2-y1)
    if dx > dy:
        n_steps = dx + 1
    else:
        n_steps = dy + 1
    
    x = np.linspace(x1, x2, n_steps, dtype=np.int32)
    y = np.linspace(y1, y2, n_steps, dtype=np.int32)
    print(x)
    print(y)

connect(0,0,3,0)
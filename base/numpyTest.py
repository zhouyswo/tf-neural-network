import numpy as np


x_data=np.float32(np.random.randint(2,100))
y_data=np.dot([0.100, 0.200], x_data) + 0.300
print(x_data,y_data)
# dataframe
import numpy as np
import pandas as pd

myarray = np.array([[1, 2, 3], [4, 5, 6]])
rownames = ['a', 'b']
colnames = ['one', 'two', 'three']

mydataframe = pd.DataFrame(myarray, index=rownames, columns=colnames)

print(mydataframe)

print("method 1:")
print("one column: %s" % mydataframe['one'])

print("method 2:")
print("one column: %s" % mydataframe.one)
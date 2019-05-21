import numpy as np
from typing import List
import cirq



X = np.asarray([[0,1],[1,0]]).reshape((2,2))
wf = np.asarray([0,1,0,0])

print(wf.shape)
print(X.reshape((2,2)))
print(wf.reshape((4,)))
print(cirq.targeted_left_multiply(X,wf,[0]).reshape((4,)))
print(cirq.targeted_left_multiply(X,wf,[1]).reshape((4,)))

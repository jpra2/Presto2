import numpy as np

a = np.array([1,2,3])
b = np.array([4,5,3])
c = np.array([4,5,7],
              [3,4,5],
              [4,5,6])

class vetor:
    def __init__(self, x, c):
        self.x = x
        self.b = c

    def produto(self):

        return np.dot(self.x, self.b)




class matrix(vetor):
    def __init__(self, c):
        super().__init__(c)


vetor1 = vetor(a)
vetor2 = vetor(b)
vetor1.produto(vetor2)

mat1 = matrix(c)
v1 = mat1.produto(vetor1)
print(v1)

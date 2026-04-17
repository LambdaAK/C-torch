from ctorch import Matrix, KNN

print(Matrix([[1, 2], [3, 4]]))
print(KNN(1, [[0, 0], [1, 1]], [[0, 1]]).predict([[0.1, 0.1]]))

from ctorch import (
    DataAugmentationType,
    GaussianNB,
    KMeans,
    KNN,
    MAB,
    Matrix,
    SVM,
    UCB,
)

print(Matrix([[1, 2], [3, 4]]))
print(KNN(1, [[0, 0], [1, 1]], [[0, 1]]).predict([[0.1, 0.1]]))

x = [[0, 0], [1, 1], [1.2, 0.8], [-0.4, -0.3]]
y01 = [[0, 1, 1, 0]]
ypm = [[-1, 1, 1, -1]]

print(GaussianNB(x, y01).predict([[0.9, 0.9]]))
print(SVM(x, ypm, learning_rate=0.0, max_iter=100, c_value=1.0, augmentation=DataAugmentationType.NO_OP).predict([[0.9, 0.9]]))
print(KMeans(2, x).assignments)

mab = MAB(3, 0.2)
arm = mab.select_arm()
mab.update(arm, 1.0)
print("mab arm:", arm)

ucb = UCB(3)
arm2 = ucb.select_arm()
ucb.update(arm2, 0.5)
print("ucb arm:", arm2)

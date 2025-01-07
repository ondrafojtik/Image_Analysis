import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
#from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib
matplotlib.use("WebAgg")

#pip install scikit-learn
#pip install matplotlib
#pip install tornado



import numpy as np
from skimage.data import lfw_subset
from skimage.feature import hog

import cv2 as cv

def hog_extract(img):
    return hog(img, orientations=8,
               pixels_per_cell=(16, 16),
               cells_per_block=(2, 2),
               visualize=False, channel_axis=None)

cv.namedWindow("img", 0)

images = lfw_subset()
cv.imshow("img", images[0])

#resized_img = cv.resize(images[120], (80, 80)
#hf = hog_extract(cv.resize(images[120], 80, 80))
hf = hog_extract(images[120]) #pocet priznaku obrazu
print("ff")
print(len(hf))

train_faces = images[:80]
print(len(train_faces))
train_faces_lab = [1] * len(train_faces)
print(train_faces_lab)

train_neg = images[100:180]
print(len(train_neg))
train_neg_lab = [0] * len(train_neg)
print(train_neg_lab)

train_all = np.concatenate((train_faces, train_neg))
train_lab = np.concatenate((train_faces_lab, train_neg_lab))

feature_list = []

print("TRAIN START")

for i, img in enumerate(train_all):
    res_img = cv.resize(img, (80, 80))
    X = hog_extract(res_img)
    print(f'feature dim: {len(X)}')
    feature_list.append(X)
    #cv.imshow("res_img", res_img)
    #cv.waitKey()

print("TRAIN END")

#FIT
clf = svm.SVC(kernel="linear")
clf.fit(feature_list, train_lab)


print("TEST START")

# 1. HOG

img_85 = cv.resize(images[85], (80, 80))
X = hog_extract(img_85)

# 2. Predict

print("TEST END")
result = clf.predict(X.reshape(1, -1))
print(f"result: {result}")

cv.imshow("img_85", img_85)
cv.waitKey()
'''

# we create 40 separable points
X, y = make_blobs(n_samples=40, centers=2, random_state=6, cluster_std=1.0)

# fit the model
clf = svm.SVC(kernel="linear")
#clf = svm.SVC(kernel="rbf")
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    plot_method="contour",
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", "--"],
    ax=ax,
)
# plot support vectors
ax.scatter(
    clf.support_vectors_[:, 0],
    clf.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)
plt.show()

'''

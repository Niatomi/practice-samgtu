import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

OBJECTS_NUMBER = 443
MINIMUM_PRECISION = 0.829
MAXIMUM_PRECISION = 0.849
FINAL_PRECISION = MINIMUM_PRECISION + (MAXIMUM_PRECISION - MINIMUM_PRECISION) / 2
TEST_SIZE = int(OBJECTS_NUMBER * 0.4)
LEGENDS = ['Группа 1', 'Группа 2', 'Группа 3', 'Группа 4']

LIGHT_AVG = 4
LIGHT_DISTORTION = 2
x_light = (LIGHT_DISTORTION / 3) * np.random.randn(OBJECTS_NUMBER) + LIGHT_AVG

WIDTH_AVG = 18
WIDTH_DISTORTION = 4
y_width = (WIDTH_DISTORTION / 3) * np.random.randn(OBJECTS_NUMBER) + WIDTH_AVG


def create_groups(baseData, distance):
    __offset = distance

    group1 = baseData
    group2 = baseData + np.array([__offset, 0])
    group3 = baseData + np.array([0, __offset])
    group4 = baseData + np.array([np.sqrt((__offset * __offset) / 2),
                                  np.sqrt((__offset * __offset) / 2)])

    return group1, group2, group3, group4


def create_dataset(group1, group2, group3, group4):
    XY = np.vstack([group1, group2, group3, group4])

    annotation1 = [0] * OBJECTS_NUMBER
    annotation2 = [1] * OBJECTS_NUMBER
    annotation3 = [2] * OBJECTS_NUMBER
    annotation4 = [3] * OBJECTS_NUMBER
    annotations = annotation1 + annotation2 + annotation3 + annotation4

    data, ident = shuffle(XY, annotations)

    x_train = data[:TEST_SIZE]
    x_test = data[TEST_SIZE:]

    y_train = ident[:TEST_SIZE]
    y_test = ident[TEST_SIZE:]


    return x_train, x_test, y_train, y_test, data, ident


def get_data():
    clf = LogisticRegression(max_iter=OBJECTS_NUMBER)
    base_set = np.stack((x_light, y_width), axis=-1)
    __i = 0
    __step = 0.01
    while True:
        __i += __step
        group_1, group_2, group_3, group_4 = create_groups(base_set, __i)
        x_train, x_test, y_train, y_test, data, ident = create_dataset(group_1, group_2, group_3, group_4)
        clf.fit(x_train, y_train)

        x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1

        x_lim = np.arange(x_min, x_max, .01)
        y_lim = np.arange(y_min, y_max, .01)

        xx, yy = np.meshgrid(x_lim, y_lim)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        result = clf.predict(x_test)
        if accuracy_score(y_test, result) >= FINAL_PRECISION:
            return group_1, group_2, group_3, group_4, xx, yy, Z, data, ident, x_train, x_test, y_train, y_test

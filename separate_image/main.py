#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import scipy.misc
from sklearn.cluster import KMeans

def __split_by_lines(image):

    # this is used to make the cv operations faster
    resize = 3.0

    rimage = scipy.misc.imresize(image, 1/resize)

    height, width = rimage.shape

    length = int(width*.55)

    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))

    # mask the structure
    contour = cv2.erode(rimage, horizontalStructure)
    contour = cv2.dilate(contour, horizontalStructure)

    # find the contours
    thresh = 30
    max_val = 255

    _, thresh = cv2.threshold(contour, thresh, max_val, cv2.THRESH_BINARY)

    _, contour, hierarchy = cv2.findContours(
        thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    splittings = []
    for cnt in contour:
        _, y, _, h = cv2.boundingRect(cnt)
        splittings.append(int((y+(h/2.0))*resize))

    return sorted(splittings)


def __split_by_whitespaces(image, n_clusters=None):

    if n_clusters is None:
        # number of clusters is not known, use simple masking

        def get_mask(image):
            height, width = image.shape
            noise_threshold_line_split = int(width*.05)

            return image.sum(axis=1) < noise_threshold_line_split

        def remove_whitespace_at_begin_and_end(image):
            mask = get_mask(image)
            for irow, row in enumerate(mask):
                if not row:
                    break
            offset = irow
            image = image[irow:, :]

            mask = get_mask(image)
            for irow, row in enumerate(mask[::-1]):
                if not row:
                    break

            return image[:len(mask)-irow, :], offset

        def get_mask_centers(image):
            mask = get_mask(image)
            whitespaces = []

            currentlyInGap = False
            current_start = None

            for ipos, pos in enumerate(mask):
                if pos:
                    if not currentlyInGap:
                        current_start = ipos
                        currentlyInGap = True
                else:
                    if currentlyInGap:
                        whitespaces.append([current_start, ipos])
                        currentlyInGap = False

            # return the centers of the whitespaces
            return map(lambda (a, b): int((a + b) * 1.0 / 2), whitespaces)

        image, offset = remove_whitespace_at_begin_and_end(image)
        centers = get_mask_centers(image)

        return sorted(map(lambda x: x+offset, centers))

    else:
        height, width = image.shape
        black_parts = np.where(image.sum(axis=1) != 0)[0]


        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(
            black_parts.reshape(black_parts.shape[0],1)
        )


        white_intervals = np.where(np.gradient(kmeans.labels_) != 0)[0]

        centers = []
        for i in range(n_clusters-1):

            centers.append(
                int((
                    black_parts[white_intervals[i*2]]
                    + black_parts[white_intervals[i*2+1]]
                )/2)
            )

        return sorted(centers)


def split(image, control_structure, orientation, output="images", n_clusters=None):

    # Check input
    if not isinstance(image, np.matrixlib.defmatrix.matrix):
        raise TypeError("image not of type numpy.matrixlib.defmatrix.matrix.")

    if control_structure not in ["blackline", "whitespace"]:
        raise ValueError("control structure has to either be 'blackline' or 'whitespace'.")

    if orientation == "columns":
        image = image.T
        # From now on we only split as rows
    elif orientation == "rows":
        pass
    else:
        raise ValueError("orientation not understood.")

    if output not in ["images", "coordinates"]:
        raise ValueError("output has to either be 'images' or 'coordinates'.")

    if control_structure == "blackline":
        splittings = __split_by_lines(image)
    else:
        splittings = __split_by_whitespaces(image, n_clusters)

    if output == "coordinates":
        return splittings
    else:
        if len(splittings) == 0:
            return [image]

        crops = []
        for i, split_pos in enumerate(splittings):
            if i == 0:
                start = 0
                end = split_pos
            else:
                start = splittings[i-1]
                end = split_pos

            crops.append(
                image[start:end, :]
            )

        # add also the last element
        crops.append(
            image[end:, :]
        )

        if orientation == "columns":
            # reflip results
            return [crop.T for crop in crops]

        return crops

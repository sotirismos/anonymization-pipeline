import os
import sys
import cv2
import numpy as np


def calculate_iou(bbox1, bbox2):
    """
    The function calculates the intersection over union of bbox1 and bbox2.

    Parameters
    ----------
    bbox1 : a list of 4 elements or a numpy array of 4 elements
        Format [top-left-x, top-left-y , bottom-right-x, bottom-right-y].
    bbox2 : a list of 4 elements or a numpy array of 4 elements
        Format [top-left-x, top-left-y , bottom-right-x, bottom-right-y].

    Returns
    -------
    iou : numpy scalar of type double
        The IoU value.
    """
    bbox1 = np.asarray(bbox1)  # ensure numpy array
    bbox2 = np.asarray(bbox2)  # ensure numpy array

    assert len(bbox1) == 4, "calculate_iou: Bbox1 length is %d" % len(bbox1)
    assert len(bbox2) == 4, "calculate_iou: Bbox2 length is %d" % len(bbox2)

    tl1 = bbox1[:2]  # top-left corner x,y pair of bbox1
    tl2 = bbox2[:2]  # top-left corner x,y pair of bbox2
    br1 = bbox1[2:]  # bottom-right corner x,y pair of bbox1
    br2 = bbox2[2:]  # bottom-right corner x,y pair of bbox2

    # width and height of the two bboxes
    wh1, wh2 = np.abs(br1 - tl1), np.abs(br2 - tl2)

    # width and height of the resulting box after the intersection
    intersection_wh = np.maximum(np.minimum(br1, br2) - np.maximum(tl1, tl2), 0.)

    # area of the resulting box after the intersection
    intersection_area = np.prod(intersection_wh)

    # the areas of the det bboxes and gt bboxes
    area_det, area_obj = (np.prod(wh1), np.prod(wh2))
    # union area
    union_area = area_det + area_obj - intersection_area

    return intersection_area / union_area


def match_gen(iou, step, n):
    """
    A generator function that yields a tuple of matching indices between the
    sets of detected bboxes and ground truth bboxes. Only non zero IoU values
    are taken into account for the actual matching.

    Parameters
    ----------
    iou : numpy array
        A numpy array that contains the IoU values 
        of the cartesian product between the detected and ground truth bbox sets.
    step : int
        The cardinality of the ground truth bboxes set.
    n : int
        The cardinality of the detected bboxes set.

    Yields
    ------
    pair : tuple
        The matcing pair of detected and ground truth bbox indices 
        (in their respective set).

    """
    iou_m = np.ma.array(iou, mask=False)  # a masked array view of the iou array
    # iou_m = np.array(iou)

    for i in range(n):
        try:  # exception handling of an out of bounds index
            iou_view = iou_m[step * i:step * (i + 1)]  # ith det bbox pairs in the cart prod set

            # gt bbox index of the pair with the max IoU among all non-zero and non-masked pairs
            # ind = iou_view.nonzero()[0][iou_view[iou_view>0].argmax()]
            ind = np.where(iou_view == max(iou_view[iou_view > 0]))[0][0]

            # mask the selected index
            # iou_m.mask[ [step*i + ind for i in range(1,n)] ] = True

            yield (i, ind)  # yield the pair (index of det bbox, index of gt bbox)
        except ValueError:
            pass
        except IndexError:
            pass


def match(det_obj, tr_bboxes):
    """
    This function matches det bboxes to gt bboxes

    Parameters
    ----------
    det_obj : ImageAI detection object
        A dictionary of the det bbox label name, condfidence value and box points.
            Key                         Value type
            ----                        ------
            name                        string
            percentage_probability      double  (range in 0-100)
            box_points                  list of uints (length 4)
    tr_bboxes : a numpy array of uints
        A 4 element array that contains a bbox top-left corner x,y and 
        bottom-right corner x,y values.

    Returns
    -------
    mpair list : a list of tuples
        A list of tuples representing the det and gt bbox index pairs.

    """
    import itertools as it
    cart_prod = it.product(det_obj, tr_bboxes)  # cartesian product of the two bbox sets

    # calculate the IoU values of the bbox pairs using list comprehension
    iou = [calculate_iou(pair[1], pair[0]['box_points']) for pair in cart_prod]

    # make an iterable
    mgen = match_gen(iou, len(tr_bboxes), len(det_obj))

    mpair_list = [mpair for mpair in mgen]

    diff = lambda a, b: np.uint(np.setdiff1d(np.union1d(a, b), np.intersect1d(a, b)))

    det_list = list()
    gt_list = list()

    if det_obj:
        if mpair_list:
            matched_det = [mpair[0] for mpair in mpair_list if mpair]
            det_list += [(ind, None) for ind in diff(matched_det, range(len(det_obj)))]
        else:
            det_list += [(ind, None) for ind in range(len(det_obj))]

    if tr_bboxes:
        if mpair_list:
            matched_gt = [mpair[1] for mpair in mpair_list if mpair]
            gt_list += [(None, ind) for ind in diff(matched_gt, range(len(tr_bboxes)))]
        else:
            gt_list += [(None, ind) for ind in range(len(tr_bboxes))]

    mpair_list += (det_list + gt_list)
    return mpair_list


def denormalize(obj, shape):
    """
    Denormalize corner coordinates of a bbox given in yolo format.

    Parameters
    ----------
    obj : a list of strings
        A list object of tokens representing a line from a yolo annotation file.
    shape : a 2 element tuple
        A tuple containing the height and width of the image 
        referenced by the annotation file.

    Raises
    ------
    Exception
        An exception is raised if the length of the object is not in accordance
        with the specified yolo annotation format.

    Returns
    -------
    bbox : numpy array of type uint
        The denormalized bbox values array.

    """
    obj = obj.split()
    shape = list(shape[::-1])  # reverse (y,x) to (x,y)
    try:
        if len(obj) != 5:
            raise Exception("Invalid annotation file, %d values where given."
                            % (len(obj.split())))
    except:
        sys.exit(-1)

    shape *= 2
    obj = obj[1:]
    denorm_obj = [float(obj[i]) * shape[i] for i in
                  range(len(obj))]  # shape[2] out of bounds or shape = (width, height, width, height)?
    denorm_obj = np.asarray(denorm_obj)

    bbox = np.zeros(len(obj), dtype='uint')
    bbox[:2] = np.round(denorm_obj[:2] - denorm_obj[2:] / 2)
    bbox[2:] = np.round(denorm_obj[:2] + denorm_obj[2:] / 2)
    return bbox


def hsv2rgb(hsv):
    import itertools as it

    select_half = lambda x: int((abs(x) % 360) // 180)
    select_sextant = lambda x: int((abs(x) % 180) // 60)

    hsv[1] = abs(hsv[1])
    hsv[-1] = abs(hsv[-1])

    hsv[1] = hsv[1] / 100 if hsv[1] > 1 else hsv[1]
    hsv[-1] = hsv[-1] / 100 if hsv[-1] > 1 else hsv[-1]

    hsv = np.asarray(hsv)
    c = np.prod(hsv[1:])
    x = c * (1 - np.abs((hsv[0] / 60) % 2 - 1))
    m = hsv[-1] - c

    tmp = [*it.permutations(np.array([c, x, 0]))]
    tmp = [tmp[::2], tmp[::-2]]

    rgb = tmp[select_half(hsv[0])][select_sextant(hsv[0])]

    return np.round(255 * (rgb + m)).astype('uint8')


def blur_img_bboxes(img_path, bboxes, out_dir):
    img = cv2.imread(img_path)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox['box_points']
        # cv2.rectangle(img, (x1, y1), (x2, y2), (128,128,128), 0)
        roi = img[y1:y2, x1:x2]
        # applying a gaussian blur over this new rectangle area
        roi = cv2.GaussianBlur(roi, (23, 23), 30)
        # impose this blurred image on original image to get final image
        img[y1:y1 + roi.shape[0], x1:x1 + roi.shape[1]] = roi

    out_path = os.path.join(out_dir,
                            os.path.split(os.path.splitext(img_path)[0])[-1] + "_det.jpg")
    cv2.imwrite(out_path, img)


def draw_img_bboxes(img_path, detections, out_dir, alpha=0.8):
    scale = lambda x: x * np.max(img.shape[0:2]) / 400

    img = cv2.imread(img_path)

    bbox_layer = img.copy()
    bbox_thickness = int(scale(1.75))

    # map detected classes to colors
    cl = list(set([det["name"] for det in detections]))

    try:
        if len(cl) == 0:
            raise ValueError
        cmap = dict((cl[i], hsv2rgb([i * 360 // len(cl) + 45, 80, 80])) for i in range(len(cl)))
    except ValueError:
        return

    for det in detections:
        class_name = det["name"]
        bbox = np.asarray(det["box_points"])
        conf = "%.2f" % (det["percentage_probability"])

        cv2.rectangle(bbox_layer, tuple(bbox[0:2]), tuple(bbox[2:]),
                      tuple(cmap[class_name].tolist()), max(bbox_thickness, 3))
        baseline = bbox[:2] - bbox_thickness * np.array([1, 2])
        txtbox_size, _ = cv2.getTextSize(class_name + ' ' + conf,
                                         cv2.FONT_HERSHEY_PLAIN,
                                         max(scale(0.4), 1),
                                         2 * max(int(scale(0.7)), 1))
        if baseline[0] + txtbox_size[0] >= img.shape[1]:
            baseline[0] = img.shape[1] - txtbox_size[0] - bbox_thickness // 2
        if baseline[1] < txtbox_size[1]:
            baseline[1] += (3 * bbox_thickness + txtbox_size[1])
            baseline[0] += 3 * (bbox_thickness // 2)

        cv2.putText(img, class_name + ' ' + conf,
                    tuple(baseline.tolist()),
                    cv2.FONT_HERSHEY_PLAIN,
                    max(scale(0.4), 1),
                    (0, 0, 0),
                    2 * max(int(scale(0.7)), 2))
        cv2.putText(img, class_name + ' ' + conf,
                    tuple(baseline.tolist()),
                    cv2.FONT_HERSHEY_PLAIN,
                    max(scale(0.4), 1),
                    (255, 255, 255),
                    max(int(scale(0.7)), 2))
        cv2.putText(bbox_layer, class_name + ' ' + conf,
                    tuple(baseline.tolist()),
                    cv2.FONT_HERSHEY_PLAIN,
                    max(scale(0.4), 1),
                    (0, 0, 0),
                    2 * max(int(scale(0.7)), 2))
        cv2.putText(bbox_layer, class_name + ' ' + conf,
                    tuple(baseline.tolist()),
                    cv2.FONT_HERSHEY_PLAIN,
                    max(scale(0.4), 1),
                    (255, 255, 255),
                    max(int(scale(0.7)), 2))

    cv2.addWeighted(bbox_layer, alpha, img, 1 - alpha, 0, img)

    out_path = os.path.join(out_dir,
                            os.path.split(os.path.splitext(img_path)[0])[-1] + "_det.jpg")
    cv2.imwrite(out_path, img)

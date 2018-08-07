import numpy as np
import glob
from skimage import measure
from skimage import filters
import matplotlib.pyplot as plt
import numpy as np
import pickle

TRUE_POSITIVE = 0
TRUE_NEGATIVE = 1
FALSE_POSITIVE = 2
FALSE_NEGATIVE = 3
#
# def get_image_classification_type(IoU_threshold, mask_real, mask_generated):
#
#     assert set(np.unique(mask_real)) <= {0, 1}
#     assert set(np.unique(mask_generated)) <= {0, 1}
#
#     if np.sum(mask_real) == 0:
#         if np.sum(mask_generated) > 0:
#             return FALSE_POSITIVE  # False Positive
#         else:
#             return TRUE_NEGATIVE  # True Negative
#
#     else:
#
#         intersection = np.sum(mask_real + mask_generated == 2)
#         union = np.sum(mask_real + mask_generated > 0)
#         IoU = intersection / union
#
#         assert IoU <= 1.0
#         if IoU < IoU_threshold:
#             return FALSE_NEGATIVE  # False Negative
#         else:
#             return TRUE_POSITIVE  # True Positive
#
# def get_score_per_threshold(IoU_threshold, softmax_threshold = 0.5):
#
#     false_negative = 0
#     false_positive = 0
#     true_negative = 0
#     true_positive = 0
#
#     for f in glob.glob('/Users/yakirgorski/Documents/projects/TGS_salt/Experiments/4/sample/train/*'):
#         image = np.load(f)
#
#         mask_real = image[:, 128:256] > 0
#         mask_generated = (1.0 - image[:, 256:]) > softmax_threshold
#
#         classification_type = get_image_classification_type(IoU_threshold, mask_real.astype(np.uint8), mask_generated.astype(np.uint8))
#
#         if classification_type == FALSE_NEGATIVE:
#             false_negative += 1
#         elif classification_type == FALSE_POSITIVE:
#             false_positive += 1
#         elif classification_type == TRUE_NEGATIVE:
#             true_negative += 1
#         elif classification_type == TRUE_POSITIVE:
#             true_positive += 1
#
#     return true_positive / (true_positive + false_positive + false_negative)
#
# score = 0.0
#
# for t in [0.5]:
#     for IoU_threshold in np.arange(0.5,1,0.05):
#         score += get_score_per_threshold(IoU_threshold, t)
#         print(t, score / 10)


def get_image_statistics(real_mask, inferred_mask, IoU_threshold):

    real_blobs = measure.label(real_mask)
    generated_blobs = measure.label(inferred_mask)

    real_blobs_indexes = np.unique(real_blobs)
    generated_blob_indexes = np.unique(generated_blobs)

    generated_blobs_statistics = FALSE_POSITIVE * np.ones_like(generated_blob_indexes)
    real_blobs_statistics = FALSE_NEGATIVE * np.ones_like(real_blobs_indexes)

    for generated_blob_idx in generated_blob_indexes[1:]:

        cur_generated_blob = generated_blobs[generated_blobs == generated_blob_idx]

        for real_blob_idx in real_blobs_indexes[1:]:

            cur_real_blob = real_blobs[real_blobs == real_blob_idx]

            intersection = np.sum(cur_generated_blob + cur_real_blob == 2)
            union = np.sum(cur_generated_blob + cur_real_blob > 0)
            IoU = intersection / union

            if IoU > IoU_threshold:
                generated_blobs_statistics[generated_blob_idx] = TRUE_POSITIVE
                real_blobs_statistics[real_blob_idx] = TRUE_POSITIVE

    FN_count = np.sum(real_blobs_statistics[1:] == FALSE_NEGATIVE)
    TP_count = np.sum(generated_blobs_statistics[1:] == TRUE_POSITIVE)
    FP_count = np.sum(generated_blobs_statistics[1:] == FALSE_POSITIVE)

    return FN_count, TP_count, FP_count


def get_IoU_score_for_threshold(IoU_threshold, softmax_threshold, beta):

    FN_count = 0
    TP_count = 0
    FP_count = 0

    for file in files:
        res = pickle.load(open(file, 'rb'))
        real_mask, inferred_mask = res[2], res[3]
        real_mask = real_mask > softmax_threshold

        #apply threshold

        cur_FN_count, cur_TP_count, cur_FP_count = get_image_statistics(real_mask, inferred_mask, IoU_threshold)

        FN_count += cur_FN_count
        TP_count += cur_TP_count
        FP_count += cur_FP_count

    F_beta = ((1 + beta ** 2) * TP_count) / (((1 + beta ** 2) * TP_count) + (beta ** 2) * FN_count + FP_count)

    return F_beta


beta = 2
score = 0.0
softmax_threshold = 0.5

IoU_thresholds = np.arange(0.5,1,0.05)
for IoU_threshold in IoU_thresholds:
    score += get_IoU_score_for_threshold(IoU_threshold, softmax_threshold, beta)
    print(score / len(IoU_thresholds))


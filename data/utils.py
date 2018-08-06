import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

PROCESSED_DATA = '/Users/yakirgorski/Documents/kaggle/Data/processed/'
TRAIN_ID_PICKLE_PATH = '/Users/yakirgorski/Documents/kaggle/Data/processed/train_IDs.pickle'
VALIDATION_ID_PICKLE_PATH = '/Users/yakirgorski/Documents/kaggle/Data/processed/validation_IDs.pickle'
TRAIN_SHIP_SEGMANTATION_PATH = os.path.join('/Users/yakirgorski/Documents/kaggle/Data', 'train_ship_segmentations.csv')
TRAIN_DIR_PATH = '/Users/yakirgorski/Documents/kaggle/Data/train'

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)


def train_val_split():

    masks = pd.read_csv(TRAIN_SHIP_SEGMANTATION_PATH)
    masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
    unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
    unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])
    # some files are too small/corrupt
    unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda c_img_id:
                                                                   os.stat(os.path.join(TRAIN_DIR_PATH,
                                                                                        c_img_id)).st_size/1024)
    unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb']>50] # keep only 50kb files
    unique_img_ids['file_size_kb'].hist()
    masks.drop(['ships'], axis=1, inplace=True)
    unique_img_ids.sample(5)

    from sklearn.model_selection import train_test_split
    train_ids, valid_ids = train_test_split(unique_img_ids,
                     test_size = 0.3,
                     stratify = unique_img_ids['ships'],
                                            random_state = 320)
    train_df = pd.merge(masks, train_ids)
    valid_df = pd.merge(masks, valid_ids)
    print(train_df.shape[0], 'training masks')
    print(valid_df.shape[0], 'validation masks')

    train_IDs = list(train_df['ImageId'].to_dict().values())
    validation_IDs = list(valid_df['ImageId'].to_dict().values())
    import pickle
    os.makedirs(PROCESSED_DATA)
    pickle.dump(train_IDs, open(TRAIN_ID_PICKLE_PATH, 'wb'))
    pickle.dump(validation_IDs, open(VALIDATION_ID_PICKLE_PATH, 'wb'))


class MaskRenderer:

    def __init__(self, masks_file_path):
        self.masks = pd.read_csv(masks_file_path)

    def get_mask(self, im_id):

        rle = self.masks.query(f'ImageId=="{im_id}"')['EncodedPixels']

        mask = masks_as_image(rle)

        return mask


if __name__ == '__main__':
    train_val_split()

# print(masks.shape[0], 'masks found')
# print(masks['ImageId'].value_counts().shape[0])
# masks.head()
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
# rle_0 = masks.query('ImageId=="00021ddc3.jpg"')['EncodedPixels']


# import os
# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from skimage.io import imread
# import matplotlib.pyplot as plt
# from skimage.segmentation import mark_boundaries
# from skimage.util.montage import montage2d as montage
# montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
# ship_dir = '../input'
# train_image_dir = os.path.join(ship_dir, 'train')
# test_image_dir = os.path.join(ship_dir, 'test')
# import gc; gc.enable() # memory is tight
#
# from skimage.morphology import label
# def multi_rle_encode(img):
#     labels = label(img[:, :, 0])
#     return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]
#
# # ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
# def rle_encode(img):
#     '''
#     img: numpy array, 1 - mask, 0 - background
#     Returns run length as string formated
#     '''
#     pixels = img.T.flatten()
#     pixels = np.concatenate([[0], pixels, [0]])
#     runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
#     runs[1::2] -= runs[::2]
#     return ' '.join(str(x) for x in runs)
#
# def rle_decode(mask_rle, shape=(768, 768)):
#     '''
#     mask_rle: run-length as string formated (start length)
#     shape: (height,width) of array to return
#     Returns numpy array, 1 - mask, 0 - background
#     '''
#     s = mask_rle.split()
#     starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
#     starts -= 1
#     ends = starts + lengths
#     img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
#     for lo, hi in zip(starts, ends):
#         img[lo:hi] = 1
#     return img.reshape(shape).T  # Needed to align to RLE direction
#
# def masks_as_image(in_mask_list):
#     # Take the individual ship masks and create a single mask array for all ships
#     all_masks = np.zeros((768, 768), dtype = np.int16)
#     #if isinstance(in_mask_list, list):
#     for mask in in_mask_list:
#         if isinstance(mask, str):
#             all_masks += rle_decode(mask)
#     return np.expand_dims(all_masks, -1)
#
# masks = pd.read_csv(os.path.join('/Users/yakirgorski/Documents/kaggle/Data', 'train_ship_segmentations.csv'))
# print(masks.shape[0], 'masks found')
# print(masks['ImageId'].value_counts().shape[0])
# masks.head()
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
# rle_0 = masks.query('ImageId=="0a1a7f395.jpg"')['EncodedPixels']
# img_0 = masks_as_image(rle_0)
# ax1.imshow(img_0[:, :, 0])
# ax1.set_title('Image$_0$')
# rle_1 = multi_rle_encode(img_0)
# img_1 = masks_as_image(rle_1)
# ax2.imshow(img_1[:, :, 0])
# ax2.set_title('Image$_1$')
# print('Check Decoding->Encoding',
#       'RLE_0:', len(rle_0), '->',
#       'RLE_1:', len(rle_1))
#
# masks = pd.read_csv(os.path.join('/Users/yakirgorski/Documents/kaggle/Data', 'train_ship_segmentations.csv'))
# masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
# unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
# unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
# unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])
# # some files are too small/corrupt
# unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda c_img_id:
#                                                                os.stat(os.path.join('/Users/yakirgorski/Documents/kaggle/Data/train',
#                                                                                     c_img_id)).st_size/1024)
# unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb']>50] # keep only 50kb files
# unique_img_ids['file_size_kb'].hist()
# masks.drop(['ships'], axis=1, inplace=True)
# unique_img_ids.sample(5)
#
# from sklearn.model_selection import train_test_split
# train_ids, valid_ids = train_test_split(unique_img_ids,
#                  test_size = 0.3,
#                  stratify = unique_img_ids['ships'])
# train_df = pd.merge(masks, train_ids)
# valid_df = pd.merge(masks, valid_ids)
# print(train_df.shape[0], 'training masks')
# print(valid_df.shape[0], 'validation masks')
#
# train_IDs = list(train_df['ImageId'].to_dict().values())
# validation_IDs = list(valid_df['ImageId'].to_dict().values())
# import pickle
# pickle.dump(train_IDs, open('/Users/yakirgorski/Documents/kaggle/Data/processed/train_IDs.pickle', 'wb'))
# pickle.dump(validation_IDs, open('/Users/yakirgorski/Documents/kaggle/Data/processed/validation_IDs.pickle', 'wb'))
#
#
#

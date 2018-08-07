import glob
import cv2
import numpy as np
import os
from .utils import MaskRenderer
from tensorpack import BatchData, AugmentImageComponents, PrefetchData
from tensorpack import imgaug
from tensorpack import RNGDataFlow
from tensorpack.utils.argtools import shape2d
from matplotlib import pyplot as plt

TRAIN_SHIP_SEGMANTATION_PATH = os.path.join('/home/paperspace/kaggle/Data/Airbus_ship/raw/all', 'train_ship_segmentations.csv')
TRAIN_DIR_PATH = '/home/paperspace/kaggle/Data/Airbus_ship/raw/all/train'

class ImageAndMaskFromFile(RNGDataFlow):
    """ Produce images read from a list of files. """
    def __init__(self, image_ids, channel=3, resize=None, shuffle=False):
        """
        Args:
            files (list): list of file paths.
            channel (int): 1 or 3. Will convert grayscale to RGB images if channel==3.
                Will produce (h, w, 1) array if channel==1.
            resize (tuple): int or (h, w) tuple. If given, resize the image.
        """
        assert len(image_ids), "No image files given to ImageFromFile!"
        self.image_ids = image_ids
        self.channel = int(channel)
        assert self.channel in [1, 3], self.channel
        self.imread_mode = cv2.IMREAD_GRAYSCALE if self.channel == 1 else cv2.IMREAD_COLOR
        if resize is not None:
            resize = shape2d(resize)
        self.resize = resize
        self.shuffle = shuffle
        self.mask_file_path = TRAIN_SHIP_SEGMANTATION_PATH
        self.images_path = TRAIN_DIR_PATH
        self.mask_renderer = MaskRenderer(masks_file_path=self.mask_file_path)


    def size(self):
        return len(self.image_ids)

    def get_data(self):
        if self.shuffle:
            self.rng.shuffle(self.image_ids)

        for im_id in self.image_ids:
            im = self._get_image(im_id)
            mask = self._get_mask(im_id)
            yield [im, mask]

    def _get_mask(self, im_id):
        return self.mask_renderer.get_mask(im_id)

    def _get_image(self, im_id):
        file_path = os.path.join(self.images_path, im_id)
        im = cv2.imread(file_path, self.imread_mode)
        assert im is not None, file_path
        if self.channel == 3:
            im = im[:, :, ::-1]
        if self.resize is not None:
            im = cv2.resize(im, tuple(self.resize[::-1]))
        if self.channel == 1:
            im = im[:, :, np.newaxis]

        return im



def get_data(image_ids, batch_size=1, is_train=False):

    ds = ImageAndMaskFromFile(image_ids, channel=3, shuffle=True)

    if is_train:

        number_of_prefetch = 8

        augs_with_label = [imgaug.RandomCrop(256),
                           imgaug.Flip(horiz=True, prob=0.5),
                           imgaug.Flip(vert=True, prob=0.5)]

        augs_no_label = [imgaug.RandomOrderAug([imgaug.Brightness(delta=20), imgaug.Contrast((0.6,1.4))])]


        # augmentors = [
        #     GoogleNetResize(),
        #     # It's OK to remove the following augs if your CPU is not fast enough.
        #     # Removing brightness/contrast/saturation does not have a significant effect on accuracy.
        #     # Removing lighting leads to a tiny drop in accuracy.
        #     imgaug.RandomOrderAug(
        #         [imgaug.BrightnessScale((0.6, 1.4), clip=False),
        #          imgaug.Contrast((0.6, 1.4), clip=False),
        #          imgaug.Saturation(0.4, rgb=False),
        #          # rgb-bgr conversion for the constants copied from fb.resnet.torch
        #          imgaug.Lighting(0.1,
        #                          eigval=np.asarray(
        #                              [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
        #                          eigvec=np.array(
        #                              [[-0.5675, 0.7192, 0.4009],
        #                               [-0.5808, -0.0045, -0.8140],
        #                               [-0.5836, -0.6948, 0.4203]],
        #                              dtype='float32')[::-1, ::-1]
        #                          )]),
        #     imgaug.Flip(horiz=True),
        # ]

    else:

        number_of_prefetch = 1

        augs_with_label = [imgaug.CenterCrop(256)]
        augs_no_label = []

    ds = AugmentImageComponents(ds, augs_with_label, (0, 1))
    ds = AugmentImageComponents(ds, augs_no_label, [0])

    ds = BatchData(ds, batch_size)
    ds = PrefetchData(ds, 30, number_of_prefetch)

    return ds

if __name__ == '__main__':
    N = 49
    image_ids = N * ['0a1a7f395.jpg'] #glob.glob('/Users/yakirgorski/Documents/kaggle/Data/train/*')[:30]
    ds = get_data(image_ids=image_ids, is_train=True).get_data()

    for i in range(N):
        image, mask = next(ds)

        plt.subplot(7,7,i+1)
        plt.imshow(image[0])

    # for i in range(N):
    #     image, mask = next(ds)
    #     plt.subplot(N,2,i*2+1)
    #     plt.imshow(image[0])
    #     plt.subplot(N,2,i*2+2)
    #     plt.imshow(mask[0][:,:,0])


    plt.show()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: Image2Image.py
# Author: Yuxin Wu

import cv2
import numpy as np
import tensorflow as tf
import glob
import os
import pickle

import argparse
import shutil
import sys
from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack import ModelDesc, SimpleTrainer
from data.dataflow import get_data
"""
To train Image-to-Image translation model with image pairs:
    ./Image2Image.py --data /path/to/datadir --mode {AtoB,BtoA}
    # datadir should contain jpg images of shpae 2s x s, formed by A and B
    # you can download some data from the original authors:
    # https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/
Training visualization will appear be in tensorboard.
To visualize on test set:
    ./Image2Image.py --sample --data /path/to/test/datadir --mode {AtoB,BtoA} --load model
"""

BATCH = 1
IN_CH = 3
OUT_CH = 1
NF = 64  # number of filter


def BNLReLU(x, name=None):
    x = BatchNorm('bn', x)
    return tf.nn.leaky_relu(x, alpha=0.2, name=name)


class Model(ModelDesc):
    def inputs(self):
        SHAPE = 256
        return [tf.placeholder(tf.float32, (None, SHAPE, SHAPE, IN_CH), 'input'),
                tf.placeholder(tf.int32, (None, SHAPE, SHAPE, OUT_CH), 'output')]

    def image2image(self, imgs):
        # imgs: input: 256x256xch
        # U-Net structure, it's slightly different from the original on the location of relu/lrelu
        with argscope(BatchNorm, training=True), \
                argscope(Dropout, is_training=True):
            # always use local stat for BN, and apply dropout even in testing
            with argscope(Conv2D, kernel_size=4, strides=2, activation=BNLReLU):
                e1 = Conv2D('conv1', imgs, NF, activation=tf.nn.leaky_relu)
                e2 = Conv2D('conv2', e1, NF * 2)
                e3 = Conv2D('conv3', e2, NF * 4)
                e4 = Conv2D('conv4', e3, NF * 8)
                e5 = Conv2D('conv5', e4, NF * 8)
                e6 = Conv2D('conv6', e5, NF * 8)
                e7 = Conv2D('conv7', e6, NF * 8)
                e8 = Conv2D('conv8', e7, NF * 8, activation=BNReLU)  # 1x1

            with argscope(Conv2DTranspose, activation=BNReLU, kernel_size=4, strides=2):
                return (LinearWrap(e8)
                        .Conv2DTranspose('deconv1', NF * 8)
                        .Dropout()
                        .ConcatWith(e7, 3)
                        .Conv2DTranspose('deconv2', NF * 8)
                        .Dropout()
                        .ConcatWith(e6, 3)
                        .Conv2DTranspose('deconv3', NF * 8)
                        .Dropout()
                        .ConcatWith(e5, 3)
                        .Conv2DTranspose('deconv4', NF * 8)
                        .ConcatWith(e4, 3)
                        .Conv2DTranspose('deconv5', NF * 4)
                        .ConcatWith(e3, 3)
                        .Conv2DTranspose('deconv6', NF * 2)
                        .ConcatWith(e2, 3)
                        .Conv2DTranspose('deconv7', NF * 1)
                        .ConcatWith(e1, 3)
                        .Conv2DTranspose('deconv8', 2, activation=tf.identity)())  # TODO: clean

    def build_graph(self, input, output):
        input = input / 128.0 - 1

        with argscope([Conv2D, Conv2DTranspose], kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)):
            fake_output = tf.identity(self.image2image(input), name='fake_output')

            softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(output), logits=fake_output), name='L1_loss')

        tf.identity(tf.nn.softmax(fake_output), name='score')
        add_moving_summary(softmax_loss)

        self.cost = softmax_loss

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=2e-4, trainable=False)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)


def sample(model_path, ds, output_path):
    pred = PredictConfig(
        session_init=get_model_loader(model_path),
        model=Model(),
        input_names=['input', 'output'],
        output_names=['input', 'output', 'fake_output', 'score'])

    pred = SimpleDatasetPredictor(pred, ds)
    for counter, o in enumerate(pred.get_result()):
        pickle.dump(o, open(os.path.join(output_path, f'{counter}.pickle')))


if __name__ == '__main__':
    TRAIN_ID_PICKLE_PATH = '/Users/yakirgorski/Documents/kaggle/Data/processed/train_IDs.pickle'
    VALIDATION_ID_PICKLE_PATH = '/Users/yakirgorski/Documents/kaggle/Data/processed/validation_IDs.pickle'

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--logger', help='logger_dir')
    parser.add_argument('--output_path', help='output_path for sampling')
    parser.add_argument('--sample', action='store_true', help='run sampling')
    parser.add_argument('--mode', choices=['AtoB', 'BtoA'], default='AtoB')
    parser.add_argument('-b', '--batch', type=int, default=1)
    global args
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    BATCH = args.batch
    validation_IDs = pickle.load(open(VALIDATION_ID_PICKLE_PATH, 'rb'))

    ds_validation = get_data(image_ids=validation_IDs, batch_size=8, is_train=False)
    if args.sample:
        assert args.load
        assert args.output_path
        os.makedirs(args.output_path)

        sample(args.load, ds_validation, args.output_path)
    else:

        logger.set_logger_dir(args.logger, action='k')
        shutil.copyfile(sys.argv[0], os.path.join(args.logger, os.path.basename(sys.argv[0])))
        train_IDs = pickle.load(open(TRAIN_ID_PICKLE_PATH, 'rb'))
        ds = get_data(image_ids=train_IDs, batch_size=args.batch, is_train=True)

        data = QueueInput(ds)
        config = AutoResumeTrainConfig(
            model=Model(),
            data=data,
            callbacks=[
                PeriodicTrigger(ModelSaver(), every_k_epochs=1),
                PeriodicTrigger(InferenceRunner(ds_validation, [ScalarStats('L1_loss')]), every_k_epochs=1),
                ScheduledHyperParamSetter('learning_rate', [(200, 1e-4)])
            ],
            steps_per_epoch=data.size(),
            max_epoch=300,
            session_init=SaverRestore(args.load) if args.load else None
        )

        launch_train_with_config(config, SimpleTrainer())
#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib
import functools
import os, sys
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from time import gmtime, strftime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import libs.configs.config_v1 as cfg
import libs.datasets.dataset_factory as datasets
import libs.nets.nets_factory as network

import libs.preprocessings.coco_v1 as coco_preprocess
import libs.nets.pyramid_network as pyramid_network
import libs.nets.resnet_v1 as resnet_v1

from train.train_utils import _configure_learning_rate, _configure_optimizer, \
    _get_variables_to_train, _get_init_fn, get_var_list_to_restore
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

resnet50 = resnet_v1.resnet_v1_50
FLAGS = tf.app.flags.FLAGS


def restore(sess):
    """choose which param to restore"""
    try:
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)
        restorer = tf.train.Saver()
        restorer.restore(sess, checkpoint_path)
        print('restored previous model %s from %s' \
              % (checkpoint_path, FLAGS.train_dir))
        time.sleep(2)
        return
    except:
        print('--restore_previous_if_exists is set, but failed to restore in %s %s' \
              % (FLAGS.train_dir, checkpoint_path))
        time.sleep(2)

    if tf.gfile.IsDirectory(FLAGS.pretrained_model):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.pretrained_model)
    else:
        checkpoint_path = FLAGS.pretrained_model

    if FLAGS.checkpoint_exclude_scopes is None:
        FLAGS.checkpoint_exclude_scopes = 'pyramid'
    if FLAGS.checkpoint_include_scopes is None:
        FLAGS.checkpoint_include_scopes = 'resnet_v1_50'

    vars_to_restore = get_var_list_to_restore()
    for var in vars_to_restore:
        print('restoring ', var.name)

    try:
        restorer = tf.train.Saver(vars_to_restore)
        restorer.restore(sess, checkpoint_path)
        print('Restored %d(%d) vars from %s' % (
            len(vars_to_restore), len(tf.global_variables()),
            checkpoint_path))
    except:
        print('Checking your params %s' % (checkpoint_path))
        raise


def train():
    """The main function that runs training"""

    ## data
    image, ih, iw, gt_boxes, gt_masks, num_instances, img_id = \
        datasets.get_dataset(FLAGS.dataset_name,
                             FLAGS.dataset_split_name,
                             FLAGS.dataset_dir,
                             FLAGS.im_batch,
                             is_training=False)
    #image_ = tf.reverse(image, axis=[-1])
    data_queue = tf.RandomShuffleQueue(capacity=32, min_after_dequeue=16,
                                       dtypes=(
                                           image.dtype, ih.dtype, iw.dtype,
                                           gt_boxes.dtype, gt_masks.dtype,
                                           num_instances.dtype, img_id.dtype))
    enqueue_op = data_queue.enqueue((image, ih, iw, gt_boxes, gt_masks, num_instances, img_id))
    data_queue_runner = tf.train.QueueRunner(data_queue, [enqueue_op] * 4)
    tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, data_queue_runner)
    (image, ih, iw, gt_boxes, gt_masks, num_instances, img_id) = data_queue.dequeue()
    im_shape = tf.shape(image)
    image = tf.reshape(image, (im_shape[0], im_shape[1], im_shape[2], 3))

    ## network
    logits, end_points, pyramid_map = network.get_network(FLAGS.network, image,
                                                          weight_decay=FLAGS.weight_decay)
    outputs = pyramid_network.build(end_points, im_shape[1], im_shape[2], pyramid_map,
                                    num_classes=81,
                                    base_anchors=9,
                                    is_training=False,
                                    gt_boxes=gt_boxes, gt_masks=gt_masks,
                                    loss_weights=[0.2, 0.2, 1.0, 0.2, 1.0])

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
            )

    sess.run(init_op)

    ## restore
    restore(sess)

    ## main loop
    coord = tf.train.Coordinator()
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

    tf.train.start_queue_runners(sess=sess, coord=coord)

    OUTPUT_PATH = './output/samples/'
    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    for step in range(10):
        o, im = sess.run([outputs, image])

        im = im[:,:,:,::-1]

        # im /= 2.
        # im += 0.5
        # im *= 255.

        boxes = o['roi']["box"]
        scores = o['roi']["score"]
        boxes_scores = zip(boxes, scores)
        boxes_scores = sorted(boxes_scores, key=lambda x: x[1], reverse=True)

        # Create figure and axes
        fig, ax = plt.subplots(2, 2)

        # Display the image
        ax[0, 0].imshow(im[0])
        ax[0, 0].set_title("All RPN BB")
        for box, score in boxes_scores:
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax[0, 0].add_patch(rect)

        ax[0, 1].imshow(im[0])
        ax[0, 1].set_title("RPN Top 10 by score")
        for box, score in boxes_scores[:10]:
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax[0, 1].add_patch(rect)

        ax[1, 0].imshow(im[0])
        ax[1, 0].set_title("Final Boxes - All")

        f_boxes = o['final_boxes']["box"]
        f_classes = o['final_boxes']["cls"]

        for box in f_boxes:
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax[1, 0].add_patch(rect)

        ax[1, 1].imshow(im[0])
        ax[1, 1].set_title("Final Boxes - Non Background")

        f_boxes = f_boxes[f_classes > 0]
        for box in f_boxes:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r',
                                     facecolor='none')

            # Add the patch to the Axes
            ax[1, 1].add_patch(rect)

        plt.savefig('%s/%d'%(OUTPUT_PATH,step))

        if coord.should_stop():
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    train()

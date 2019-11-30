from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np

from official.utils.arg_parsers import parsers
from official.utils.logging import hooks_helper

import dan_model

def dan_model_fn(features,
                 groundtruth,
                 mode,
                 stage,
                 num_lmark,
                 model_class,
                 mean_shape,
                 imgs_mean,
                 imgs_std,
                 data_format, multi_gpu=False):

    if isinstance(features, dict):
        features = features['image']

    model = model_class(num_lmark,data_format)
    resultdict = model(features,
                       stage==1 and mode==tf.estimator.ModeKeys.TRAIN, #False
                       stage==2 and mode==tf.estimator.ModeKeys.TRAIN, #False
                       mean_shape,imgs_mean,imgs_std)                  #None, None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=resultdict)

def dan_main(flags, model_function):
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True)
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9,
                                                    session_config=session_config)
    estimator = tf.estimator.Estimator(
        model_fn=model_function, model_dir=flags["model_dir"], config=run_config,
        params={
                'dan_stage':flags["dan_stage"],
                'num_lmark':flags["num_lmark"],
                'data_format': flags["data_format"],
                'batch_size': flags["batch_size"],
                'multi_gpu': False,
            })

    return estimator

    
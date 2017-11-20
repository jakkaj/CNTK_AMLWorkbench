# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os, sys
import numpy as np
import utils.od_utils as od
from utils.config_helpers import merge_configs
from cntk import load_model
def get_configuration(detector_name):
    # load configs for detector, base network and data set
    if detector_name == "FastRCNN":
        from FastRCNN.FastRCNN_config import cfg as detector_cfg
    elif detector_name == "FasterRCNN":
        from FasterRCNN.FasterRCNN_config import cfg as detector_cfg
    else:
        print('Unknown detector: {}'.format(detector_name))

    # for VGG16 base model use:         from utils.configs.VGG16_config import cfg as network_cfg
    # for AlexNet base model use:       from utils.configs.AlexNet_config import cfg as network_cfg
    from utils.configs.AlexNet_config import cfg as network_cfg
    # for Pascal VOC 2007 data set use: from utils.configs.Pascal_config import cfg as dataset_cfg
    # for the Grocery data set use:     from utils.configs.Grocery_config import cfg as dataset_cfg
    from utils.configs.Animals_config import cfg as dataset_cfg

    return merge_configs([detector_cfg, network_cfg, dataset_cfg, {'DETECTOR': detector_name}])

def do_detect(img_path):
    

    detector_name = "FasterRCNN"
    cfg = get_configuration(detector_name)

    # train and test
    cfg['MODEL_PATH'] = "C:/Users/jak/azureml/share/jordoexperiments/JordanMLWorkspace/TechSummit2017ML/faster_rcnn_eval_AlexNet_e2e.model"
    eval_model = od.train_object_detector(cfg)
    #eval_model = load_model("C:/Users/jak/Source/Repos/TechSummit 2017/PretainedModels/faster_rcnn_eval_AlexNet_e2e.model")
    # eval_results = od.evaluate_test_set(eval_model, cfg)    

    regressed_rois, cls_probs = od.evaluate_single_image(eval_model, img_path, cfg)
    bboxes, labels, scores = od.filter_results(regressed_rois, cls_probs, cfg)

    # write detection results to output
    fg_boxes = np.where(labels > 0)
    print("#bboxes: before nms: {}, after nms: {}, foreground: {}".format(len(regressed_rois), len(bboxes), len(fg_boxes[0])))
    for i in fg_boxes[0]: print("{:<12} (label: {:<2}), score: {:.3f}, box: {}".format(
                                cfg["DATA"].CLASSES[labels[i]], labels[i], scores[i], [int(v) for v in bboxes[i]]))

    # visualize detections on image
    return od.visualize_results(img_path, bboxes, labels, scores, cfg)



    # measure inference time
    #od.measure_inference_time(eval_model, img_path, cfg, num_repetitions=100)

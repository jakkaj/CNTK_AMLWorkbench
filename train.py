# Spark configuration and packages specification. The dependencies defined in
# this file will be automatically provisioned for each run that uses Spark.

import argparse

import os, sys
import numpy as np
import utils.od_utils as od
from utils.config_helpers import merge_configs
from cntk import load_model
available_detectors = ['FastRCNN', 'FasterRCNN']
import dl



#from azureml.logging import get_azureml_logger

# initialize the logger
#logger = get_azureml_logger()

# add experiment arguments
#parser = argparse.ArgumentParser()
# parser.add_argument('--arg', action='store_true', help='My Arg')
#args = parser.parse_args()
#print(args)

# This is how you log scalar metrics
#logger.log("RunTest", 1.1)

# Create the outputs folder - save any outputs you want managed by AzureML here
os.makedirs('./outputs', exist_ok=True)

def get_detector_name(args):
    detector_name = None
    default_detector = 'FasterRCNN'
    if len(args) != 2:
        print("Please provide a detector name as the single argument. Usage:")
        print("    python DetectionDemo.py <detector_name>")
        print("Available detectors: {}".format(available_detectors))
    else:
        detector_name = args[1]
        if not any(detector_name == x for x in available_detectors):
            print("Unknown detector: {}.".format(detector_name))
            print("Available detectors: {}".format(available_detectors))
            detector_name = None

    if detector_name is None:
        print("Using default detector: {}".format(default_detector))
        return default_detector
    else:
        return detector_name

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

if __name__ == '__main__':
    # Currently supported detectors: 'FastRCNN', 'FasterRCNN'
    
    dl.downloadModel()

    args = sys.argv
    detector_name = get_detector_name(args)
    cfg = get_configuration(detector_name)

    cfg["SAVE_PATH"] = os.path.abspath(os.path.join(".", "outputs", "animals.model"));

    # train and test
    eval_model = od.train_object_detector(cfg)
    eval_results = od.evaluate_test_set(eval_model, cfg)

    #savePath = os.path.join(osy.path.dirname(os.path.abspath(__file__)), r"output.model")
    #eval_model.save(savePath)

    
    # write AP results to output
    for class_name in eval_results: print('AP for {:>15} = {:.4f}'.format(class_name, eval_results[class_name]))
    print('Mean AP = {:.4f}'.format(np.nanmean(list(eval_results.values()))))
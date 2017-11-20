# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import zipfile
import os, sys



if __name__ == '__main__':
    base_folder = os.path.dirname(os.path.abspath(__file__))



    sys.path.append(os.path.join(base_folder, "PretrainedModels"))
    from download_model import download_model_by_name
    download_model_by_name("AlexNet_ImageNet_Caffe")

    print("Creating mapping files for Grocery data set..")
    create_grocery_mappings(base_folder)

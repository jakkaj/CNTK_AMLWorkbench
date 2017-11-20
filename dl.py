import os
import sys
try:
    from urllib.request import urlretrieve 
except ImportError: 
    from urllib import urlretrieve

#https://www.cntk.ai/Models/Caffe_Converted/AlexNet_ImageNet_Caffe.model
def downloadModel():
    
    filename=os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'] + "AlexNet_ImageNet_Caffe.model"
    
    print("Looking for model: " + filename)

    if not os.path.exists(filename):
        print("dl alexnet model")
        urlretrieve("https://www.cntk.ai/Models/Caffe_Converted/AlexNet_ImageNet_Caffe.model", filename)
        print("done download model")
    else:
        print("Model present")
if __name__ == '__main__':downloadModel()
    
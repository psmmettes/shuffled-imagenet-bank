#
# Obtain the representation of a video frame using our Shuffled ImageNet
# pre-trained models.
#
# Pascal Mettes (2019).
#
# Please cite the following paper when using the pre-trained models:
#
#@article{mettes2020shuffled,
#  title={Shuffled ImageNet-Banks for Video Event Detection and Search},
#  author={Mettes, Pascal and Koelma, Dennis C and Snoek, Cees G M},
#  journal={Transactions on Multimedia Computing Communications and Applications},
#  year={2020},
#  publisher={ACM}
#}

import os
import sys
import mxnet as mx
import numpy as np
from collections import namedtuple
from nltk.corpus import wordnet as wn

#
# Load and pre-process a single image from file.
#
# Input
# imagename : File name of the frame.
#
# Output
# img       : Processed image.
#
def get_image(imagename):
    img = mx.image.imread(imagename)
    img = mx.image.imresize(img, 224, 224)
    img = img.transpose((2, 0, 1))
    img = img.expand_dims(axis=0)
    img = img.astype('float32')
    return img

#
# Extract features from a specified layer for a single frame.
#
# Input
# mod       : MxNet model.
# imagename : File name of the frame.
# layer     : Specified output layer (prob or fc).
#
# Output
# features  : Vector representation of the frame.
#
def extract_features(mod, imagename, layer="prob"):
    Batch = namedtuple('Batch', ['data'])
    image = get_image(imagename)
    mod.forward(Batch([image]))
    features = mod.get_outputs()[0].asnumpy()[0,:]
    return features

#
# Only perform extraction when directly calling the script.
#
if __name__ == "__main__":
    # The user specifies frame filename and which output layer to use.
    imagename = sys.argv[1]
    modeldir  = sys.argv[2]
    layer     = sys.argv[3]
    assert(os.path.exists(imagename) and (layer == "prob" or layer == "fc"))

    # Set the context on GPU.
    ctx = mx.gpu()
    
    # Load the model.
    prefix = modeldir + "resnext-101"
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, 38)
    if layer == "prob":
        mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        allow_missing = True
    elif layer == "fc":
        internals = sym.get_internals()
        #print(internals)
        #exit()
        fsym = internals["flatten0_output"]
        mod = mx.mod.Module(symbol=fsym, context=ctx, label_names=None)
        allow_missing = False
    mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], 
         label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=allow_missing)
    
    # Perform feature extraction.
    features = extract_features(mod, imagename, layer)
    print("Obtained features:", features.shape)
    
    # If features are the probabilities, show the scores and concepts.
    if layer == "prob" and len(sys.argv) == 5:
        wnidfile = sys.argv[4]
        concepts = [line.strip() for line in open(wnidfile)]
        concepts = np.array(concepts)
        # Print top 5 concepts with scores.
        porder = np.argsort(features)[::-1]
        print("Top 5 concepts with scores:")
        for i in range(5):
            cid   = concepts[porder[i]]
            cname = wn.synset_from_pos_and_offset('n',int(cid[1:]))
            print("%s : %.4f" %(cname, features[porder[i]]))

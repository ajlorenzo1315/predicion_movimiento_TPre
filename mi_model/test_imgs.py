# librerias basicas
import torch, torchvision
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets


import mmdet
from mmcv.ops import get_compiling_cuda_version, get_compiler_version

# librerias para el detector de mascaras
import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
from mmdet.datasets.pipelines import Compose


#librerias de fichero
import os, sys, errno

import glob
import argparse

# librerias para trar con la imagen
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import cv2
from collections import OrderedDict

# tiempo 
from timer import Timer


class Me_model():
    
    def __init__ (self,detector,device=None):
        
        torch.cuda.empty_cache() 
        # Cargamos los modelos necesarios

        # Set the device to be used for evaluation
        
        if device:
            self.device=device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Load the config
        config = mmcv.Config.fromfile(detector.config)

        self.model_name=detector.config.split("/")[-1].split(".")[0]
        # Set pretrained to be None since we do not need pretrained model here
        config.model.pretrained = None

        # Initialize the detector
        self.model = build_detector(config.model)

        # Load checkpoint
        checkpoint = load_checkpoint(self.model,detector.checkpoint, map_location=self.device)

        # Set the classes of models for inference
        self.model.CLASSES = checkpoint['meta']['CLASSES']

        # We need to set the model's cfg for inference
        self.model.cfg = config

        # Convert the model to GPU
        self.model.to(self.device)

        # Convert the model into evaluation mode
        self.model.eval()

    
    def mask(self,path,paths_out="out",ext="png",save=True):
        
        paths= sorted(glob.glob(os.path.join(path, '*.{}'.format(ext))))
        os.makedirs(paths_out, exist_ok=True)
        if save:
            self.fichero = open(paths_out+'/mi_fichero', 'w')
        else:
            self.fichero = None
        timer = Timer()
        for idx, image_path in enumerate(paths):# Use the detector to do inference
        
                
                
                img= cv2.imread(image_path)

                self.shape=img.shape

                timer.tic()
                result=inference_detector(self.model, img)
                timer.toc()

                text_scale = 1
                text_thickness = 2
                line_thickness = 3

                fps=1. / timer.average_time

                cv2.putText(img, '%s ' % (self.model_name) ,
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0,0, 255), thickness=2)
                cv2.putText(img, 'frame: %d fps: %.2f' % (idx, fps),
                    (0, int(30* text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)
                    
                output_name = os.path.splitext(os.path.basename(image_path))[0]
                name_dest_npy = os.path.join(paths_out,self.model_name,"{}.png".format(output_name))
                show_result_pyplot(self.model, img, result, score_thr=0.3,out_file=name_dest_npy)
                    
                print(" foto ",idx,"fps", 1. / max(1e-5, timer.average_time))
               
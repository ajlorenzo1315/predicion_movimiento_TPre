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


# profundidad
from models.model import GLPDepth
#tracker
from tracker.byte_tracker import BYTETracker


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
    
    def __init__ (self,detector,depth,tracker,device=None):
        
        torch.cuda.empty_cache() 
        # Cargamos los modelos necesarios

        # Set the device to be used for evaluation
        
        if device:
            self.device=device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Load the config
        config = mmcv.Config.fromfile(detector.config)
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


        self.depth = GLPDepth(max_depth=10, is_train=False).to(self.device)
        depth_weight = torch.load(depth.checkpoint)
        if 'module' in next(iter(depth_weight.items()))[0]:
                depth_weight = OrderedDict((k[7:], v) for k, v in depth_weight.items())
        self.depth.load_state_dict(depth_weight)
        self.depth.eval()

        self.tracker = BYTETracker(tracker, frame_rate=tracker.frame_rate)
        self.ags_tracker=tracker
        self.shape=(480,640)

    def plot_tracking(self,image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
        im = np.ascontiguousarray(np.copy(image))
        im_h, im_w = im.shape[:2]

        top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

        #text_scale = max(1, image.shape[1] / 1600.)
        #text_thickness = 2
        #line_thickness = max(1, int(image.shape[1] / 500.))
        text_scale = 2
        text_thickness = 2
        line_thickness = 3

        radius = max(5, int(im_w/140.))
        cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                    (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

        for i, tlwh in enumerate(tlwhs):
            x1, y1, w, h = tlwh
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            obj_id = int(obj_ids[i])
            id_text = '{}'.format(int(obj_id))
            if ids2 is not None:
                id_text = id_text + ', {}'.format(int(ids2[i]))
            color = self.get_color(abs(obj_id))
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
            cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                        thickness=text_thickness)
        return im

    def pos_x_y_new(self,boxs,ids,masks,depth,depths,frame,view=True):
        if view:

            depth1= cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY )
            imagen_depth=np.zeros((depth1.shape))
            image_depth_human=np.zeros((depth.shape))
            imagen_view_new=np.zeros((depth.shape))
            self.shape=depth1.shape

        puntos=[]

        cont_nop=0
    
        for idx_box in range(len(boxs)):
            if view:
                mascara=masks[idx_box]*255
                mascara=np.uint8(mascara)

                imagen_unida=  cv2.bitwise_and(depth1, depth1, mask=mascara)
                imagen_depth+=imagen_unida
                
            
            depth= depths[idx_box]
        
            
            puntos.append((int( boxs[idx_box][0]+boxs[idx_box][2]//2) , int(depth*self.shape[0]/255) ) )
            if view:
                str_save="\t".join([str(frame),str(ids[idx_box]),str(puntos[idx_box][0]),str(puntos[idx_box][1])])
                self.fichero.write(str_save + os.linesep)
            if view:
                cv2.circle(imagen_view_new,(puntos[idx_box][0],puntos[idx_box][1]),10,(self.get_color(ids[idx_box])),-1)
          
            
        if view:
            image_depth_human[:,:,0]=imagen_depth
            image_depth_human[:,:,1]=imagen_depth
            image_depth_human[:,:,2]=imagen_depth
            
            return np.concatenate([image_depth_human,imagen_view_new], 1),puntos

        else:
            return puntos

    
    def change_view(self,path,paths_out="out",ext="png",save=True):
        
        paths= sorted(glob.glob(os.path.join(path, '*.{}'.format(ext))))
        os.makedirs(paths_out, exist_ok=True)
        if save:
            self.fichero = open(paths_out+'/mi_fichero', 'w')
        timer = Timer()
        for idx, image_path in enumerate(paths):# Use the detector to do inference
        
                timer.tic()
                
                img= cv2.imread(image_path)

                self.shape=img.shape
                if idx%1==0:
                    result=inference_detector(self.model, img)
                        
                    bboxes,masks=self.gef_get_box([result[0][0],[result[1][0]]])
                    
                    #profundidad en los pixeles pares
                
                    im = pil.fromarray(img)

                    original_width, original_height= im.size

                    input_image = transforms.ToTensor()(im).unsqueeze(0)
                    input_image = input_image.to(self.device)
                    outputs = self.depth(input_image)
                

                    disp = outputs["pred_d"]
                    disp_resized = torch.nn.functional.interpolate(
                    disp, (original_height, original_width), mode="bilinear", align_corners=False)


                    output_name = os.path.splitext(os.path.basename(image_path))[0]


                    name_dest_npy = os.path.join(paths_out, "{}.png".format(output_name))
                    
                    disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
                
                    imga=np.ones((disp_resized_np.shape))*disp_resized_np.max()
                    disp_resized_np=imga-disp_resized_np
                    vmax = np.percentile(disp_resized_np, 95)
                    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                    

                if bboxes is not None:
                        if idx%1==0:
                            online_targets = self.tracker.update(bboxes,masks ,[img.shape[0],img.shape[1]], [img.shape[0],img.shape[1]],
                                                            colormapped_im)
                        else:
                            #actualizamos la profundida con el mismo valor
                            online_targets = self.tracker.update(bboxes,masks ,[img.shape[0],img.shape[1]], [img.shape[0],img.shape[1]],
                            update_depth=True)

                        online_tlwhs = []
                        online_mask = []
                        online_ids = []
                        online_scores = []
                        online_depth=[]
                        for t in online_targets:
                                tlwh = t.tlwh
                                
                                tid = t.track_id
                                vertical = tlwh[2] / tlwh[3] > self.ags_tracker.aspect_ratio_thresh
                                if tlwh[2] * tlwh[3] > self.ags_tracker.min_box_area and not vertical:
                                    online_tlwhs.append(tlwh)
                                    online_ids.append(tid)
                                    online_scores.append(t.score)
                                    online_mask.append(t.mask)
                                    online_depth.append(t.depth)
                                    

                        timer.toc()

                        
                           
                        image_depth_human=self.pos_x_y_new(online_tlwhs,online_ids,online_mask,colormapped_im,online_depth,idx,save)

                        if len(image_depth_human)>1:
                            image_depth_human,puntos=image_depth_human
                            image_depth_human=self.plot_tracking(
                                image_depth_human, online_tlwhs, online_ids, frame_id= idx + 1, fps=1. / timer.average_time)

                        else:
                            puntos=image_depth_human
                            image_depth_human=np.zeros((img.shape[0],img.shape[1]*2,img.shape[2]))
                        if save:
                            online_im = self.plot_tracking(
                                    img, online_tlwhs, online_ids, frame_id= idx + 1, fps=1. / timer.average_time)

                else:  
                    online_im = img
                    image_depth_human=np.zeros((img.shape[0],img.shape[1]*2,img.shape[2]))
                    timer.toc()

                if save:
                    colormapped_im2=self.plot_tracking(
                                    colormapped_im.copy(), online_tlwhs, online_ids, frame_id= idx + 1, fps=1. / timer.average_time)

                    online_im=np.concatenate([online_im,colormapped_im2], 1)   

                    online_im=np.concatenate([online_im,image_depth_human], 0)  
                    
                    cv2.imwrite(name_dest_npy, online_im)
                   
                print(" foto ",idx,"fps", 1. / max(1e-5, timer.average_time))
    
        self.fichero.close()
    @staticmethod
    def gef_get_box(result):
        bbox_result,mask_result = result
        bboxes = np.vstack(bbox_result)
        scores = bboxes[:,4]
       
        if len(result) == 0:
            bboxes = np.zeros([0, 5])
            masks = np.zeros([0, 0, 0])
     
        else:
            masks = mmcv.concat_list(mask_result)

            if isinstance(masks[0], torch.Tensor):
                masks = torch.stack(masks, dim=0).detach().cpu().numpy()
            else:
                masks = np.stack(masks, axis=0)
            
            if bboxes[:, :4].sum() == 0:
                num_masks = len(bboxes)
                x_any = masks.any(axis=1)
                y_any = masks.any(axis=2)
                for idx in range(num_masks):
                    x = np.where(x_any[idx, :])[0]
                    y = np.where(y_any[idx, :])[0]
                    if len(x) > 0 and len(y) > 0:
                        bboxes[idx, :4] = np.around( np.array(
                            [x[0], y[0], x[-1] + 1, y[-1] + 1],
                            dtype=np.float32) )
                        
        return  bboxes,masks 
    
    @staticmethod
    def get_color(idx):
        idx = idx * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

        return color
    
    

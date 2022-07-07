'''
Doyeon Kim, 2022
'''

import os
import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm


import cv2
import numpy as np
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

import utils.logging as logging
import utils.metrics as metrics
from models.model import GLPDepth

from timer import Timer
import argparse

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)

    parser.add_argument('--out_path', default="results/out", type=str,
                        help='path to a test image or folder of images')

    parser.add_argument('--ckpt_dir', type=str,
                        help='name of a pretrained model to use',
                       )
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")

    # depth configs
    parser.add_argument('--max_depth',      type=float, default=10.0)
    parser.add_argument('--max_depth_eval', type=float, default=10.0)
    parser.add_argument('--min_depth_eval', type=float, default=1e-3)        
    parser.add_argument('--do_kb_crop',     type=int, default=1)
    parser.add_argument('--kitti_crop', type=str, default=None,
                            choices=['garg_crop', 'eigen_crop'])

    return parser.parse_args()



def main():
    # experiments setting
    args = parse_args()


    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("MODEL RUN IN :",device)
  
    model = GLPDepth(max_depth=args.max_depth, is_train=False).to(device)
    model_weight = torch.load(args.ckpt_dir)
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight)
    model.eval()

    print("\n3. Inference & Evaluate")
     # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.out_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.out_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))
    # create output folder
    os.makedirs(output_directory, exist_ok=True)
    timer = Timer()
    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')

            original_width, original_height = input_image.size
            imgae_save_to=input_image
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            timer.tic()
            input_image = input_image.to(device)
            
            outputs = model(input_image)
            #print(outputs)
            disp = outputs["pred_d"]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)
            timer.toc()
            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            #name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            #scaled_disp, depth_resized = disp_to_depth(disp, 0.1, 100)
            #np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            #print(disp_resized_np.shape)

            imga=np.ones((disp_resized_np.shape))*disp_resized_np.max()
            disp_resized_np=imga-disp_resized_np
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            #print(input_rgb.shape)
            #img_neo=np.concatenate((input_rgb,colormapped_im))
            text_scale = 2
            text_thickness = 2
            line_thickness = 1
            time_avg=timer.average_time
            if time_avg==0:
                time_avg=1
            fps=max(1e-10, 1. / time_avg)
            cv2.putText(colormapped_im, 'fps: %.2f' % fps,
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness=2)


            im = pil.fromarray(colormapped_im)
            #image_00004900_0.png_disp.jpeg
            name_foto=image_path.split("/")
            name_foto=name_foto[-1].split(".")
            name_foto=name_foto[0]
            name_dest_im = os.path.join('results',"{}_disp.jpeg".format(name_foto))
            
            # concatenate both vertically
           
            name_deth_pro = os.path.join(output_directory,"{}.png".format(name_foto))
            
            im.save(name_deth_pro)
            #name_corped_rgb = os.path.join('results',"{}rgb.png".format(name_foto))
            #im_grey.save(name_grey_depth) 
            #input_r.save(name_corped_rgb)
            
            #just save a single depth
            #print("len",im.shape,input_r.shape)
            #im.save(name_dest_im)

            #save a concatenated iamge for depth and rgb
            #imwrite(name_dest_im,image[:,:,::-1])
            
            print("   Processed {:d} in fps {:f}  of {:d} images - saved prediction to {} ".format(
                idx + 1, fps,len(paths), name_dest_im))

    print('-> Done!')
   


if __name__ == "__main__":
    main()

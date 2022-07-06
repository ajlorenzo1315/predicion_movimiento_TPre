# predicion_movimiento_TPre

# Deteccion
## OpenMMLab website mmdetection [[Ghithub]](https://github.com/open-mmlab/mmdetection)
[![PyPI](https://img.shields.io/pypi/v/mmdet)](https://pypi.org/project/mmdet)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmdetection.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmdetection/workflows/build/badge.svg)](https://github.com/open-mmlab/mmdetection/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmdetection/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmdetection)
[![license](https://img.shields.io/github/license/open-mmlab/mmdetection.svg)](https://github.com/open-mmlab/mmdetection/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmdetection.svg)](https://github.com/open-mmlab/mmdetection/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmdetection.svg)](https://github.com/open-mmlab/mmdetection/issues)

[📘Documentation](https://mmdetection.readthedocs.io/en/stable/) |
[🛠️Installation](https://mmdetection.readthedocs.io/en/stable/get_started.html) |
[👀Model Zoo](https://mmdetection.readthedocs.io/en/stable/model_zoo.html) |
[🆕Update News](https://mmdetection.readthedocs.io/en/stable/changelog.html) |
[🚀Ongoing Projects](https://github.com/open-mmlab/mmdetection/projects) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmdetection/issues/new/choose)

### SoloV2 [[Paper]](https://arxiv.org/abs/2003.10152) [[Github]](https://github.com/open-mmlab/mmdetection/edit/master/configs/solov2)
<img src="assets/SOLOV2_arq.png" width="400"/>   <img src="assets/SOLOV2_arq_2.png" width="400"/>

#### SoloV2
|  Backbone  |  Style  | MS train | Lr schd | Mem (GB) | mask AP |                                                    Config                                                     |                                                                                                                                                Download                                                                                                                                                |
| :--------: | :-----: | :------: | :-----: | :------: | :-----: | :-----------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|    R-50    | pytorch |    N     |   1x    |   5.1    |  34.8   |   [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/solov2/solov2_r50_fpn_1x_coco.py)    |      [model](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r50_fpn_1x_coco/solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth)           \| [log](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r50_fpn_1x_coco/solov2_r50_fpn_1x_coco_20220512_125858.log.json)      |
|    R-50    | pytorch |    Y     |   3x    |   5.1    |  37.5   |   [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/solov2/solov2_r50_fpn_3x_coco.py)    |      [model](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r50_fpn_3x_coco/solov2_r50_fpn_3x_coco_20220512_125856-fed092d4.pth)           \| [log](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r50_fpn_3x_coco/solov2_r50_fpn_3x_coco_20220512_125856.log.json)      |
|   R-101    | pytorch |    Y     |   3x    |   6.9    |  39.1   |   [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/solov2/solov2_r101_fpn_3x_coco.py)   |     [model](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r101_fpn_3x_coco/solov2_r101_fpn_3x_coco_20220511_095119-c559a076.pth)         \| [log](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r101_fpn_3x_coco/solov2_r101_fpn_3x_coco_20220511_095119.log.json)     |
| R-101(DCN) | pytorch |    Y     |   3x    |   7.1    |  41.2   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/solov2/solov2_r101_dcn_fpn_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r101_dcn_fpn_3x_coco/solov2_r101_dcn_fpn_3x_coco_20220513_214734-16c966cb.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r101_dcn_fpn_3x_coco/solov2_r101_dcn_fpn_3x_coco_20220513_214734.log.json) |
| X-101(DCN) | pytorch |    Y     |   3x    |   11.3   |  42.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/solov2/solov2_x101_dcn_fpn_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_x101_dcn_fpn_3x_coco/solov2_x101_dcn_fpn_3x_coco_20220513_214337-aef41095.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_x101_dcn_fpn_3x_coco/solov2_x101_dcn_fpn_3x_coco_20220513_214337.log.json) |

#### Light SOLOv2

| Backbone |  Style  | MS train | Lr schd | Mem (GB) | mask AP |                                                     Config                                                     |                                                                                                                                                  Download                                                                                                                                                  |
| :------: | :-----: | :------: | :-----: | :------: | :-----: | :------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-18   | pytorch |    Y     |   3x    |   9.1    |  29.7   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/solov2/solov2_light_r18_fpn_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_light_r18_fpn_3x_coco/solov2_light_r18_fpn_3x_coco_20220511_083717-75fa355b.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_light_r18_fpn_3x_coco/solov2_light_r18_fpn_3x_coco_20220511_083717.log.json) |
|   R-34   | pytorch |    Y     |   3x    |   9.3    |  31.9   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/solov2/solov2_light_r34_fpn_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_light_r34_fpn_3x_coco/solov2_light_r34_fpn_3x_coco_20220511_091839-e51659d3.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_light_r34_fpn_3x_coco/solov2_light_r34_fpn_3x_coco_20220511_091839.log.json) |
|   R-50   | pytorch |    Y     |   3x    |   9.9    |  33.7   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/solov2/solov2_light_r50_fpn_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_light_r50_fpn_3x_coco/solov2_light_r50_fpn_3x_coco_20220512_165256-c93a6074.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_light_r50_fpn_3x_coco/solov2_light_r50_fpn_3x_coco_20220512_165256.log.json) |



# Seguimiento

## ByteTrack [[paper]](https://arxiv.org/abs/2110.06864)[[Ghithub]](https://github.com/ifzhang/ByteTrack)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bytetrack-multi-object-tracking-by-1/multi-object-tracking-on-mot17)](https://paperswithcode.com/sota/multi-object-tracking-on-mot17?p=bytetrack-multi-object-tracking-by-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bytetrack-multi-object-tracking-by-1/multi-object-tracking-on-mot20-1)](https://paperswithcode.com/sota/multi-object-tracking-on-mot20-1?p=bytetrack-multi-object-tracking-by-1)

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bDilg4cmXFa8HCKHbsZ_p16p0vrhLyu0?usp=sharing)

<p align="center"><img src="assets/teasing.png" width="400"/></p>


### Test en MOT
#### Resultados en el conjunto de prueba de desafío MOT
| Dataset    |  MOTA | IDF1 | HOTA | MT | ML | FP | FN | IDs | FPS |
|------------|-------|------|------|-------|-------|------|------|------|------|
|MOT17       | 80.3 | 77.3 | 63.1 | 53.2% | 14.5% | 25491 | 83721 | 2196 | 29.6 |
|MOT20       | 77.8 | 75.2 | 61.3 | 69.2% | 9.5%  | 26249 | 87594 | 1223 | 13.7 |

#### Resultados de visualización en el conjunto de prueba de desafío MOT
<img src="assets/MOT17-01-SDP.gif" width="400"/>   <img src="assets/MOT17-07-SDP.gif" width="400"/>
<img src="assets/MOT20-07.gif" width="400"/>   <img src="assets/MOT20-08.gif" width="400"/>



# Depth
## Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth [[Paper]](https://arxiv.org/abs/2201.07436) [[Ghithub]](https://github.com/vinvino02/GLPDepth)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/global-local-path-networks-for-monocular/monocular-depth-estimation-on-nyu-depth-v2)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=global-local-path-networks-for-monocular)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/global-local-path-networks-for-monocular/monocular-depth-estimation-on-kitti-eigen)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen?p=global-local-path-networks-for-monocular)

### arquitectura

<p align="center"><img src="assets/dethp_arq.png" width="400"/></p>

### Test KITTI
<p align="center"><img src="assets/detph_kitti.png" width="400"/></p>

### Resultados de visualización 

<p align="center"><img src="assets/dethp_nyu.png" width="400"/></p>


### Downloads
- [[Downloads]](https://drive.google.com/drive/folders/17yYbLZS2uQ6UVn5ET9RhVL0y_X3Ipl5_?usp=sharing) Trained ckpt files for NYU Depth V2 and KITTI
- [[Downloads]](https://drive.google.com/drive/folders/1LGNSKSaXguLTuCJ3Ay_UsYC188JNCK-j?usp=sharing) Predicted depth maps png files for NYU Depth V2 and KITTI Eigen split test set


### Google Colab

<p>
<a href="https://colab.research.google.com/drive/1v6fzr4XusKdXAaeGZ1gKe1kh9Ce_WQhl?usp=sharing" target="_parent">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
</p>
Thanks for the great Colab demo from NielsRogge

# Prediccion

# Instalaciones 

Siga las guias de cada uno de los apartados que aporntan en su github

# Training

Habria que entrenear cada parte 

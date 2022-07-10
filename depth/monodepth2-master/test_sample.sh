python test_simple.py --out_path results/mono+stereo_1024x320  --model_name mono+stereo_640x192 --ext png   --image_path ./assets/lp-left --pred_metric_depth
python test_simple.py --out_path results/mono_640x192 --model_name mono_640x192 --ext png   --image_path ./assets/lp-left --pred_metric_depth
python test_simple.py --out_path results/stereo_640x192 --model_name stereo_640x192 --ext png   --image_path ./assets/lp-left --pred_metric_depth
python test_simple.py --out_path results/mono_1024x320 --model_name mono_1024x320 --ext png   --image_path ./assets/lp-left --pred_metric_depth
python test_simple.py --out_path results/stereo_1024x320 --model_name stereo_1024x320 --ext png   --image_path ./assets/lp-left --pred_metric_depth
python test_simple.py --out_path results/mono+stereo_1024x320 --model_name mono+stereo_1024x320 --ext png   --image_path ./assets/lp-left --pred_metric_depth

# Environment and Supported Toolkits

 python 3.9<br>
 pytorch(http://pytorch.org/)<br>
 tensorflow 2.10.0<br>
 munch 2.5.0<br>
 opencv-python 4.4.0.46<br>
 ffmpeg-python 0.2.0<br>
 
# Demo

 1. Download pre-trained models from [BaiduNetdisk](https://pan.baidu.com/s/1JI9dT4wWasm8_A56pQHXqA). password: wy12.<br>
 2. Create the folder expr, which contains the folders :checkpoints, results, samples.
 3. Copy the pre-training files to the expr/checkpoints/BraTS2018.
 4. To train ADGAN, run the following command：<br>
```bash
  #BraTS2018
   python main.py --mode train --num_domains 4 --w_hpf 0 \
               --lambda_reg 1 --lambda_sty 1 --lambda_per 1 --lambda_l1 1 \
               --train_img_dir data/BraTS2018/train \
               --val_img_dir data/BraTS2018/val
```
 5. Test ADGAN by running the following command：<br>
```bash
 #BraTS2018
 python main.py --mode sample --num_domains 4 --resume_iter 100000 --w_hpf 0 \
               --checkpoint_dir expr/checkpoints/BraTS2018 \
               --result_dir expr/results/BraTS2018 \
               --src_dir assets/representative/BraTS2018/src \
               --ref_dir assets/representative/BraTS2018/ref
```
# Notes
1. The implementation of proposed ADGAN model is based on StarGAN V2 (https://github.com/clovaai/stargan-v2). 
2. To facilitate processing, some image data were uploaded, which were derived from the dataset BraTS2018.
3. For smooth training of the network, it is recommended that the image naming does not contain any modal nouns.

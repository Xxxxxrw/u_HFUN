# u-HFUN: Ultra-Lightweight Network with Hierarchical Feature Upsampler for Single Image Super-Resolution

This is an official implementation of the paper “u-HFUN: Ultra-Lightweight Network with Hierarchical Feature Upsampler for Single Image Super-Resolution”.

## Dependencies

- The cuda version used in the project is 12.2
- Python = 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
```
conda create -n your_env_name python=3.8
conda activate your_env_name
```
- Other packages required by the project are in the file ‘requirements.txt’
```
pip install -r requirements.txt
```
# Model framework 
1. The network model implementation code (6 blocks and 10 blocks) is located in directory `./src/model/`
2. Network model diagram：
<img width="717" height="224" alt="image" src="https://github.com/user-attachments/assets/7a905676-2d27-4e37-bbfb-761fc545d825" />
<img width="742" height="272" alt="image" src="https://github.com/user-attachments/assets/f8923e2e-3455-4221-b8f8-4253d04441f2" />


# Codes 
- This repository provides the code for training and testing.
  
## How to Test
1. Download the five test datasets (Set5, Set14, B100, Urban100, Manga109) from [Google Drive](https://drive.google.com/drive/folders/1lsoyAjsUEyp7gm1t6vZI9j7jr9YzKzcF?usp=sharing)

2. All versions of pretrained models have be placed in `./pre_models/` folder. 

3. The testing commands are placed in the './src/demo.sh' file. 
Close comments in 'demo.sh' and run 'demo.sh' to execute the corresponding command of testing. Such as:
```
python main.py --model u_hfun --save ./test/u_HFUN_Div2k_tiny_x2 --scale 2 --n_feats 32 --pre_train /Your_Path/u_HFUN/pretrained/../ --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
```

4. More testing commonds can be found in `./src/demo.sh` file and the output results will be sorted in `./experiment/test/`

## How to Train

1. Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) from [Here](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) 

3. Modify the training dataset path attributes '--dir_data' and '--data_train' in the `./src/option.py` file.

4. The training commands are placed in the './src/demo.sh' file.
Close comments in 'demo.sh' and run 'demo.sh' to execute the corresponding command of training. Such as:
```
python main.py --model u_hfun --save ./train/u_HFUN_Div2k_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
```
4. More training commond can be found in `./src/demo.sh` file, and the training results will be sorted in `./experiment/train/`

## Network quantitative comparison
<img width="585" height="628" alt="image" src="https://github.com/user-attachments/assets/812c0c83-a411-49e0-8444-c63df46afc4e" />


## SR images visualization
1. We provided visualization of SR images of the model from [Google Drive](https://drive.google.com/drive/folders/1xiPOE22AExEcIe5-er3clOYFHCVCJo6F?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1vEOJaLGScgRGaIOeFI3q8w) (code：s8eg)
2. Visual comparison of reconstructions in the paper（u_HFUN）:
<img width="592" height="499" alt="image" src="https://github.com/user-attachments/assets/618dc5ac-41bb-4c38-8eb7-1045ff363581" />
<img width="589" height="501" alt="image" src="https://github.com/user-attachments/assets/529a8d48-32a8-41b8-8d42-f93138f9c4da" />



## Contact
Email: fjs1867@mnnu.edu.cn


If you find our work is useful, please kindly cite it.
```
Upload in the future
```

## License
This project is released under the Apache 2.0 license.


## Acknowledgements
This code is built on [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch). Thanks for the awesome work.

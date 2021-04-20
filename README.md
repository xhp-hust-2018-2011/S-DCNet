# S-DCNet
This is the repository for S-DCNet, presented in our paper in the ICCV 2019:

[**From Open Set to Closed Set: Supervised Spatial Divide-and-Conquer for Object Counting**](https://arxiv.org/pdf/2001.01886.pdf)

[Haipeng Xiong](https://scholar.google.com/citations?user=AEW8GxcAAAAJ&hl=zh-CN)<sup>1</sup>, Hao Lu<sup>2</sup>, Chengxin Liu<sup>1</sup>,
Liang Liu<sup>1</sup>, Zhiguo Cao<sup>1</sup>, Chunhua Shen<sup>2</sup>

<sup>1</sup>Huazhong University of Science and Technology, China

<sup>2</sup>The University of Adelaide, Australia

## News !!!
An extended version of S-DCNet, i.e., [SS-DCNet](https://github.com/xhp-hust-2018-2011/SS-DCNet) is now available !!! 

## Contributions
- **Reformulating the counting problem:** We propose S-DCNet, which transforms open-set counting into a closed-set problem via Spatial Divide-and-Conquer;
- **Simple and effective:** S-DCNet achieves the state-of-the-art performance on three crowd counting datasets (ShanghaiTech, UCF_CC_50 and UCF-QNRF), a vehicle counting dataset (TRANCOS) and a plant counting dataset (MTC). Compared to the previous best methods, S-DCNet brings a 20.2% relative improvement on the ShanghaiTech Part_B, 20.9% on the UCF-QNRF, 22.5% on the TRANCOS and 15.1% on the MTC.

## Environment
Please install required packages according to `requirements.txt`.

## Data
Testing data for ShanghaiTech dataset have been preprocessed. You can download the processed dataset from:

[Baidu Yun (314M)](https://pan.baidu.com/s/1lSqT7_9wCR4xW-rd4gyPpg) with code: ou3b

[Google Drive (314M)](https://drive.google.com/file/d/10lGXpMHfNTa-y_5IN6V2HrjqI39f2Bac/view?usp=sharing)

## Model
Pretrained weights can be downloaded from:

[Baidu Yun (210MB)](https://pan.baidu.com/s/1yIyjqdM594Q0Tdw0oBq8_w) with code: 1tcb

[Google Drive (210MB)](https://drive.google.com/file/d/1vaJsll0K1vkK2keKJoGy18--1j63U27l/view?usp=sharing)

## A Quick Demo
1. Download the code, data and model.

2. Organize them into one folder. The final path structure looks like this:
```
-->The whole project
    -->Test_Data
        -->SH_partA_Density_map
        -->SH_partB_Density_map
    -->model
        -->SHA
        -->SHB
    -->Network
        -->class_func.py
        -->merge_func.py
        -->SDCNet.py
    -->SHAB_main.py
    -->main_process.py
    -->Val.py
    -->load_data_V2.py
    -->IOtools.py
```

3. Run the following code to reproduce our results. The MAE will be SHA: 57.575, SHB: 6.633. Have fun:)
    
       python SHAB_main.py


## References
If you find this work or code useful for your research, please cite:
```
@inproceedings{xhp2019SDCNet,
  title={From Open Set to Closed Set: Counting Objects by Spatial Divide-and-Conquer},
  author={Xiong, Haipeng and Lu, Hao and Liu, Chengxin and Liang, Liu and Cao, Zhiguo and Shen, Chunhua},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2019},
  pages = {8362-8371}
}
```

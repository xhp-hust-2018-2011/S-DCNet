# S-DCNet
This is the repository for S-DCNet, presented in our paper in the ICCV 2019:

**From Open Set to Closed Set: Counting Objects by Spatial Divide-and-Conquer** [[paper]](https://arxiv.org/pdf/1908.06473.pdf)

Haipeng Xiong, [Hao Lu](https://sites.google.com/site/poppinace/)<sup>2</sup>, Chengxin Liu<sup>1</sup>,
Liang Liu<sup>1</sup>, Zhiguo Cao<sup>1</sup>, [Chunhua Shen](http://cs.adelaide.edu.au/~chhshen/)<sup>2</sup>

<sup>1</sup>Huazhong University of Science and Technology, China

<sup>2</sup>The University of Adelaide, Australia

## Contributions
- **Reformulate counting problem:** We propose S-DCNet, which transforms open-set counting into a closed-set problem via Spatial Divide-and-Conquer;
- **Simple and effective:** S-DCNet achieves the state-of-the-art performance on three crowd counting datasets (ShanghaiTech, UCF_CC_50 and UCF-QNRF),  a vehicle counting dataset (TRANCOS) and a plant counting dataset (MTC). Compared to the previous best methods, S-DCNet brings a 20.2% relative improvement on the ShanghaiTech Part_B, 20.9% on the UCF-QNRF, 22.5% on the TRANCOS and 15.1% on the MTC.

## Environment
- python 3.6
- pytorch 0.4.0 or higher version
- numpy 1.14.0
- scikit-image 0.13.1
- scipy 1.0.0
- pandas 0.22.0

## Data
Test data for ShanghaiTech dataset has been processed, you can download the processed dataset from [SHTech_Data](https://pan.baidu.com/s/1lSqT7_9wCR4xW-rd4gyPpg) with code: ou3b.

## Model
Trained models can be down load from [SHTech_Model](https://pan.baidu.com/s/1yIyjqdM594Q0Tdw0oBq8_w) with code:1tcb.

## A Quick Demo
1. Download the code, data and model. 
2. Then put them in one folder, and the final path structure will look like this:
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

3. Run the code and have fun.
> python SHAB_main.py



## References
```
@inproceedings{xhp2019SDCNet,
  title={From Open Set to Closed Set: Counting Objects by Spatial Divide-and-Conquer},
  author={Xiong, Haipeng and Lu, Hao and Liu, Chengxin and Liang, Liu and Cao, Zhiguo and Shen, Chunhua},
  booktitle={Proceedings of the IEEE Conference on Computer Vision (ICCV)},
  year={2019}
}
```

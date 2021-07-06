# U-RISC-Data-Code
This repo holds data and code for U-RISC cell membrane segmentation.

## Dataset

### 1. Download

There are three ways for users to download the dataset.

* Google Drive


 > https://drive.google.com/drive/folders/14u-TMK8kDJWMvZbr7KTQKwHpFqe9lrYU?usp=sharing


* MEGA 

 > https://mega.nz/folder/HTIBkQzb#Dg2duJyMlghT9u5tM6EH7Q

* Baidu Cloud

 > https://pan.baidu.com/s/11WRPCJOu-R1h5ns3Fht80A code：zciq 




### 2. Folder Directory

* U-RISC-DATA
    - imgs
        - Track1
            - train
            - val
            - test
        - Track2
        - Track2\_original\_size
    - label
    - Human\_Annotation\_Process
        - 1st
        - 2nd
        - final


## Evaluation

The output of U-RISC dataset are binary segmentation images. And the evaluation criterion is F1-score. 

### Examples:

* Evaluate on Single Image:

	```
	cd Evaluation
	python eval.py --evalon1pic True --pre_path ./pre.png --gt_path ./gt.png
	```

* Evaluate on Forder Image:

	```
	cd Evaluation
	python eval.py --evalon1pic False --pre_path ./pres --gt_path ./gts
	```

## Segmentation Networks

This part holds code for segmentation networks of U-RISC cell membrane segmentation. 

***Updating...***


## Citations

```
@article{shi2021u,
  title={U-RISC: An Ultra-high Resolution EM Dataset Challenging Existing Deep Learning Algorithms},
  author={Shi, Ruohua and Wang, Wenyao and Li, Zhixuan and He, Liuyuan and Sheng, Kaiwen and Ma, Lei and Du, Kai and Jiang, Tingting and Huang, Tiejun},
  journal={bioRxiv},
  year={2021},
}

```


	
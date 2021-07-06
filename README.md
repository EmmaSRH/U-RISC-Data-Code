# U-RISC-Data-Code
The data and code of U-RISC cell membrane segmentation

## Dataset

### Download

Please download the dataset from 

### Folder Directory

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

### Hardware
NVIDIA A100 (GPU Memory 48GB each)

### Packages
> pip install -r requirements.txt

### U-Net
* train

	```
	cd U-Net
	CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py	```
	
* test

	```
	cd U-Net
	CUDA_VISIBLE_DEVICES=0 python test.py
	```


### LinkNet
* train

	```
	cd LinkNet/src
	CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py	```
	
* test

	```
	cd LinkNet/inference
	CUDA_VISIBLE_DEVICES=0 python test.py
	```
	
### CASENet
* train

	```
	cd CASENet
	CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py	```
	
* test

	```
	cd CASENet
	CUDA_VISIBLE_DEVICES=0 python test.py
	```

### U-Net-transfer
* train

	```
	cd U-Net-transfer
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py	```
	
* test

	```
	cd U-Net-transfer
	CUDA_VISIBLE_DEVICES=0 python test.py
	```
	
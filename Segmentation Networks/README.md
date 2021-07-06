# U-RISC-Data-Code

## Segmentation Networks

### Hardware
NVIDIA A100 (GPU Memory 48GB each)

### Packages
> pip install -r requirements.txt

### U-Net
* train

	```
	cd U-Net
	CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py	
	

* test

	```
	cd U-Net
	CUDA_VISIBLE_DEVICES=0 python test.py
	```


### LinkNet
* train

	```
	cd LinkNet/src
	CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py	
	
* test

	```
	cd LinkNet/inference
	CUDA_VISIBLE_DEVICES=0 python test.py
	```
	
### CASENet
* train

	```
	cd CASENet
	CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py
	
* test

	```
	cd CASENet
	CUDA_VISIBLE_DEVICES=0 python test.py
	```

### U-Net-transfer
* train

	```
	cd U-Net-transfer
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py	
	
* test

	```
	cd U-Net-transfer
	CUDA_VISIBLE_DEVICES=0 python test.py
	```
	
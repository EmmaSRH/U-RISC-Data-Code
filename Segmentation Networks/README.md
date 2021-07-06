# U-RISC-Data-Code

## Segmentation Networks

### 1. Hardware
NVIDIA A100 (GPU Memory 48GB each)

### 2. Packages
> pip install -r requirements.txt

### 3. U-Net
* train

	```
	cd U-Net
	CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py	
	

* test

	```
	cd U-Net
	CUDA_VISIBLE_DEVICES=0 python test.py
	```


### 4. LinkNet
* train

	```
	cd LinkNet/src
	CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py	
	
* test

	```
	cd LinkNet/inference
	CUDA_VISIBLE_DEVICES=0 python test.py
	```
	
### 5. CASENet
* train

	```
	cd CASENet
	CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py
	
* test

	```
	cd CASENet
	CUDA_VISIBLE_DEVICES=0 python test.py
	```

### 6. U-Net-transfer
* train

	```
	cd U-Net-transfer
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py	
	
* test

	```
	cd U-Net-transfer
	CUDA_VISIBLE_DEVICES=0 python test.py
	```

### 7. Weight

The weight for networks can be download from

* Google Drive

 > https://drive.google.com/drive/folders/1SJdsfoNUNJmBVcT_UFb_KgNsVum_uCmi?usp=sharing


* MEGA 

 > https://mega.nz/folder/HDw3QAYa#bR333xgA0cSA69UaFfMjig
 
* Baidu Cloud

 > https://pan.baidu.com/s/1VkQ5W_h2e2mGLtrgTosBjA code：o9h3 
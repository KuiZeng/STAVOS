# Project Introduction
This project is divided into two main parts, 
STAVOS_ Main is responsible for training the model, including the code for the semi supervised video object segmentation model, The STAVOS model can be used to segment general videos. 
STAVOS_ Medaka is responsible for the application of Medaka, which can segment the heartbeat video of Medaka, obtain its ventricular related parameters and visualization results.


# Display of project results
![3D_spheroids](https://github.com/KuiZeng/STAVOS/assets/139167726/4739e30b-ed4a-414a-9df5-428dd313f7a2)  ![electrocardiogram](https://github.com/KuiZeng/STAVOS/assets/139167726/b3a1650b-aab5-4d2c-8007-a1ad74d4af0f)
![visual_00037](https://github.com/KuiZeng/STAVOS/assets/139167726/75dcfa4e-06e3-492e-b0cd-505f9456b46e)  ![visual_00041](https://github.com/KuiZeng/STAVOS/assets/139167726/26c0e11b-b147-4ed9-950e-584d87e68c69)
![visual_00000](https://github.com/KuiZeng/STAVOS/assets/139167726/dd0debd6-fc0e-4e89-b5f0-cfea658537f9)  ![visual_00012](https://github.com/KuiZeng/STAVOS/assets/139167726/2a20d1c4-9d31-4f42-afe6-3f13290d302a)

# folder introduce：
![image](https://github.com/KuiZeng/STAVOS/assets/139167726/df05a3d6-5cdb-499b-84b3-52ee1f3ba62b)
##### Data-medaka-Lateral-right folder: N datasets
##### Data-medaka-Ventural folder: R dataset
##### DAVIS-2016-trainval:
##### DAVIS-2017-trainval-480p:
##### input folder:the video to be inputted
##### output folder: The output result of the automation system
##### STAVOS_main:project code for training and testing the weights of the STAVOS model
##### STAVOS_medaka: automation system code for medaka ventricle
##### test_predict: Test results of the dataset
##### trained_model: Weights of trained models


# data acquisition
Due to the upper limit on the file size uploaded by GitHub, the dataset and model weights cannot be uploaded here.


##### Data-medaka-Lateral-right folder: https://cowtransfer.com/s/d2546440518240
##### Data-medaka-Ventural folder: https://cowtransfer.com/s/3d0ed4b3d8ae4b
##### trained_model: https://cowtransfer.com/s/7c37a69ceedf4d
##### DAVIS-2016-trainval:https://davischallenge.org.
##### DAVIS-2017-trainval-480p:https://davischallenge.org.


# How to use:
## 1, Download the required dataset and project code
## 2, configuration environment
  ##### Deep learning environment：conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
  ##### Other libraries: Install whatever is missing.
## 3, Ensure the relative position of files
  ##### If not correct, follow the code prompts to make corrections
## 4, training and testing
  ##### --STAVOS_main
  #####   run "train_main.py" for model training
  #####   run "test_main.py" for model testing
  ##### You can also download the trained model from https://cowtransfer.com/s/6ec386cb595042, or contact email:2514819977@qq.com
## 5, application
  ##### --STAVOS_medaka
  #####   run "main.py" for medaka ventricular segmentation and parameter acquisition
    

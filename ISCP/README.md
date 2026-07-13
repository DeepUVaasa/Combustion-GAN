# Combustion-GAN
Abnormal combustion cycle detection 

# Dataset 
## The preprocessed in-cylinder pressure data can be downloaded from this google drive link:

https://drive.google.com/file/d/1CUlUp6NBntQr_viBEE4XilfZsM2aILJI/view?usp=sharing (contain preprocessed data (3 data splits))

Download the data and unzip them in ./experiments/iscp/dataset/ folder. <br> 
dataset folder shoud be:<br> 
dataset/ <br>
        final_augmented_combustion_pressure_data_w20_s20_split1.npz <br>
        final_augmented_combustion_pressure_data_w20_s20_split2.npz <br>
        final_augmented_combustion_pressure_data_w20_s20_split3.npz <br>
        
# Trained Model
The trained model is provided in the ./experiment/iscp/output/CombustionGAN/iscp/model/ folder.

# Evaluation 
For evaluation, open run_iscp.sh file and change the value of "test=1" if it is not already 1. 

``
bash run_iscp.sh 
``

The reconstructed samples can be found in ./experiments/iscp/output/CombustionGAN/iscp/test/ folder. The reconstruction error between the input test data and the reconstructed data will be in the same folder.  

# Training 
If you want to train your own model, open the run_iscp.sh file and replace the "test=1" to "test=0". Save the file and run the following command in the terminal:

``
bash run_iscp.sh 
``

# Acknowledgement 
This code is based on BeatGAN (https://github.com/imbingox/BeatGAN), thanks for this wonderful work.

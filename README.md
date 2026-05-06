# Combustion-GAN
Abnormal combustion cycle classification 

# ECG Dataset
Download the ECG (full MIT-BIH dataset) dataset from BeatGAN github:  

https://github.com/imbingox/BeatGAN 

place the dataset in experiments/ecg/dataset/preprocessed/


# Training and Evaluation
To train the model, open run_ecg.sh and update run_ecg.sh file (test=0).

``
bash run_ecg.sh 
``

To evaluate on the test data after training, update the run_ecg.sh file (test=1)

``
bash run_ecg.sh 
``

# Acknowledgement 
We want to the BeatGAN authors for their code.

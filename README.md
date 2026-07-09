# Environment Setup 
If you already have an Ubuntu machine (Ubuntu version >= 20.4), and the Anaconda (https://www.anaconda.com/download) is installed in it, do the following to setup the enviromnent: 

To create a new environment in conda, run the following command in the terminal 
``
conda create -n combustiongan python==3.13 
``
To activate the environment, run the following command in the terminal 
``
conda activate combustiongan 
``
# Packages Installation 
From pytorch (https://pytorch.org/) website, install the latest version of pytorch. Additionally install the follow python packages: 

``
pip install numpy matplotlib seaborn scipy scikit-learn scikit-image tqdm 
``
For Electrocardiography (ecg) data, go to ECG folder. 

For in-cylinder combustion pressure (ISCP) data, go to ISCP folder. 

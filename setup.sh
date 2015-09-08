sudo apt-get install -y gcc g++ gfortran build-essential git wget linux-image-generic libopenblas-dev python-dev python-pip python-nose python-numpy python-scipy
sudo pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git 
sudo pip install lasagne
sudo wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb 
sudo dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb  
sudo apt-get update
sudo apt-get install -y cuda 
echo -e "\nexport PATH=/usr/local/cuda/bin:$PATH\n\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64" >> .bashrc 
sudo reboot


sudo apt-get update
sudo apt-get -y dist-upgrade

cuda-install-samples-7.0.sh ~/

sudo apt-get install nvidia-340
sudo apt-get install nvidia-346-vm
cd NVIDIA_CUDA-7.0_Samples/
cd 1_Utilities/deviceQuery
make
./deviceQuery
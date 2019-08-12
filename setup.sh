#!/usr/bin/env bash

export LC_ALL=C

apt-get -y install python3-pip
apt-get -y install unzip

#apt-get -y install libsm6 libxrender-dev libxrender1 libfontconfig1
#pip3 install opencv-python
#python3 -m pip install opencv-contrib-python

pip3 install tqdm;
pip3 install imageio;
pip3 install comet_ml;
pip3 install matplotlib
#pip3 install bs4
#pip3 install requests
#apt-get -y install python3-tk;

wget https://gist.githubusercontent.com/schmidtdominik/4d520346c6e5e528f51b332bb7bb8788/raw/0a332b580098c84003370fcdab2afc575252e3ff/dl_from_gdrive.py;
python3 dl_from_gdrive.py 1yeLkE1p5oeCqa5pA7tc0tI4eoyvjZc5X mars32k.zip
unzip mars32k.zip
mkdir Datasets
mv mars32k Datasets/mars32k


# pip install jupyter;
# jupyter notebook --ip=127.0.0.1 --port=8080 --allow-root
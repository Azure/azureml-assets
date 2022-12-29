#!/bin/sh -x
cwd="$(pwd)"
echo "azcopy installation started ...."
echo "Downloading azcopy to file azcopy.tar ...."
wget https://aka.ms/downloadazcopy-v10-linux -O azcopy.tar --no-verbose
tar -xvf azcopy.tar
mkdir ./bin # store azcopy executable here
echo "copying azcopy bin file to ./bin/"
cp ./azcopy_linux_amd64_*/azcopy ./bin/
echo "setting azcopy path"
dir=$cwd/bin
export PATH=$PATH:$dir
echo "execute azcopy -h"
azcopy -h
echo "cleaning downloaded files"
rm azcopy.tar
rm -rf ./azcopy_linux_amd64_*
echo "azcopy installation completed ...."

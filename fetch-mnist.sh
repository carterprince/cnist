#!/bin/bash

mkdir -p mnist
cd mnist

# Train
wget https://web.archive.org/web/20200515032833if_/http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget https://web.archive.org/web/20200513204951if_/http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz

# Test
wget https://web.archive.org/web/20200802112939if_/http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget https://web.archive.org/web/20200513191053if_/http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz

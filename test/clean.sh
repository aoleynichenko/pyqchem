#!/usr/bin/sh

for d in `ls -d */`;
do
    echo $d
    cd $d
    rm output *.{dat,npy}
    cd ..
    echo '============================'
done;

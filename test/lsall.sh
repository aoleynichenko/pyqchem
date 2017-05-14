#!/usr/bin/sh

for d in `ls -d */`;
do
    echo $d
    cd $d
    ls
    cd ..
    echo '============================'
done;

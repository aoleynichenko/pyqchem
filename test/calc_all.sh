#!/usr/bin/sh

alias pyqchem='python ~/gitdir/pyqchem/pyqchem.py'

for d in `ls -d */`;
do
    echo $d
    cd $d
    
    if [ ! -f t2.npy ]; then
        pyqchem | tee output
    fi;
    
    cd ..
    echo '============================'
done;

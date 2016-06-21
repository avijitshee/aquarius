#!/bin/bash
for ext in `ls`; do
    if [ -f $ext/.git.bak ]; then
        echo "Updating $ext"
        cd $ext
        git pull
        cd ..
    fi
done

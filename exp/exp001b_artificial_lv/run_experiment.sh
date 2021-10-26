#!/bin/sh

case "$1" in
    0)  lotkavolterra-run @common_args.txt \
            --logdir "./run_$1"
        ;;
    *)  echo "This sub-experiment does not exist yet";;
esac
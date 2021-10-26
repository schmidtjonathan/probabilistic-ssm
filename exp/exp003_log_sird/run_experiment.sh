#!/bin/sh

case "$1" in
    "ger")  log-sird-run @common_args.txt \
            --logdir "./run_$1" \
            --country "Germany"
        ;;
    *)  echo "This sub-experiment does not exist yet";;
esac
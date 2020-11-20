#!/bin/bash

dev_dir="."

while getopts d: flag
do
    case "${flag}" in
        d) devdir=${OPTARG};;
    esac
done

echo "Using Development Directory $devdir";

cd "$dev_dir";

echo "Checking for minichess...";
[ -d "minichess" ] && echo "Found minichess. Pulling..." && git pull || echo "Cloning minichess..." && git clone https://github.com/mdhiebert/minichess.git
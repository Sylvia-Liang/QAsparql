#!/bin/bash
set -e


echo "Downloading Glove"
cd glove/
wget -q -c http://www-nlp.stanford.edu/data/glove.840B.300d.zip
unzip -q glove.840B.300d.zip

mkdir -p models/
cd models/
wget -r -nH --cut-dirs=3 --no-parent --reject="index.html*" http://isis-data.science.uva.nl/mettes/imagenet-shuffle/mxnet/
cd ..

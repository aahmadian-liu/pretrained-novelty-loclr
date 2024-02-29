# Novelty Detection in Pretrained Representation Space with Locally Adapted Likelihood Ratio

To run novelty detection experiments using Python, please:

1. Set the dataset paths in *datasets.py*. To use iBOT, you also need to download the ViT-B/16 ImageNet-1K pretrained backbone model from https://github.com/bytedance/ibot#pre-trained-models (name it as 'model.pth').

2. Use *representations.py* to extract representations of images and normal/novel splits for a dataset, and store them in the 'workspace' directory.

3. Run a novelty detection method with appropriate command line arguments, e.g., using *vglr.py*, *knn.py*.


A suitable Python environment is specified in *pip_requirements.txt* .

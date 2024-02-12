# Readme

To run novelty detection experiments:

1. Set the dataset paths in 'datasets.py'. To use iBOT, you also need to download the ViT-B/16 ImageNet-1K pretrained backbone model from https://github.com/bytedance/ibot#pre-trained-models (name as 'model.pth')

2. Use 'representations.py' to extract representations of images and normal/novel splits for a dataset, and store them in 'workspace'

3. Run the novelty detection methods, e.g., using 'vglr.py', 'knn.py'.

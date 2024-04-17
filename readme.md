# Requirements
The entire code is in Python 3 (tested with version 3.9). The following packages are required to extract representations and run novelty detection experiments. The number in parenthesis shows the version which we have tested with.

- numpy (1.21.0)
- scikit-learn (1.0.2)
- faiss-gpu (1.7.2)
- torch (1.13.1)
- torchvision (0.14.1)
- scipy (1.7.3)
- timm (0.9.8)
- matplotlib (3.5)
- pillow (9.3.0)
- tqdm

# Obtaining data representations
- The DINO ImageNet pretrained models should be downloaded automatically via torch hub. If you would like to use a pretrained iBOT ViT model, please download the corresponding backbone checkpoint file from the [official iBOT repository](https://github.com/bytedance/ibot), and save it as 'checkpoint_ibot.pth' in the code directory.

- Please look at the `dataloading.py` module, and specify the paths where the files for each dataset are (either automatically or manually) downloaded to. 

- The data representation files will always be written to and read from the workspace directory, with the default path **'./workspace'**. You can change this path by setting the environment variable `ood_ws`. 

- Run the `representations.py` script with appropriate parameters to extract representation vectors of dataset images, and divide them into normal (in-distribution) and novel (out-of-distribution) sets. You can see explanations of the input arguments by running `python representations.py --help`.

    Examples:

    `python representations.py flowers flowers_dino --classes_in 0:51 --classes_out 51:102`

    `python representations.py cub cub_ibot_r0 --classes_in rand_0:100 --classes_out rand_100:200 --model_name ibot_vitb16 --randseed 0`

    `python representations.py pcam pcam_dino --classes_in 0:1 --classes_out 1:2`
    


# Running novelty detection methods
- Run one of the scripts `vglr.py`, `knn.py`, `nnofnn.py`, `osvm.py`, `kmeansmaha.py`, or `localknfst.py` to experiment with the corresponding novelty detection methods. The first input argument always refers to the name of file containing the data representations in the workspace (name without extension, i.e. same as passed to the second argument of `representations.py`). Each method might require additional parameters to be specified. Please run a script with `--help` to see more details.

    Examples:

    `python vglr.py flowers_dino 10`

    `python knn.py pcam_dino 1`

- The results will be saved in **'./output'** directory as human readable text. The output includes the performance numbers (AUROC and FPR@95TPR) as well as the values of used arguments. By default, this is written/appended to a text file named by date, and marked with method-date-time tags. The file path can be changed by providing the '--output_file' argument when running a method.    

# Reference repositories
Parts of the following repositories have been used in our implementation:

- https://github.com/facebookresearch/dino
- https://github.com/bytedance/ibot
- https://github.com/stanislavfort/exploring_the_limits_of_OOD_detection
- https://github.com/cvjena/knfst 
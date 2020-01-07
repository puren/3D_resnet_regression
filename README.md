This is a pytorch implementation for joint state estimation from RGB, depth and RGB-Depth image using 3D-resnet architecture presented in the paper [Hara et al, "Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?", 2018](http://openaccess.thecvf.com/content_cvpr_2018/html/Hara_Can_Spatiotemporal_3D_CVPR_2018_paper.html). In this paper, they show that 3D CNN architectures can be trained from scratch using big video datasets for action recognition. In this code, we convert this architectures to regression architecture to estimate joint states of robotic manipulator directly from video images.
 
The network model implementations are copied from the repository [3D-ResNets-Pytorch](https://github.com/kenshohara/3D-ResNets-PyTorch) and are altered. Please check [the paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Hara_Can_Spatiotemporal_3D_CVPR_2018_paper.html) and [the repository](https://github.com/kenshohara/3D-ResNets-PyTorch) for more detailed explanations about the method, results and implementation details.

# To run the code:
In "data" folder, there is a small example dataset to demonstrate what kind of data the code uses and how to create a dataset.
Steps to run the code
1. Create .csv files that contains information of train, validation and test splits of the dataset:

```python
python createCSV.py --data_dir data/ --perc_train 50 --perc_test 25 --per_val 25.
```

In the "data/" folder, where the dataset exits, this code will create {train, val, test}.csv files and info_csv.xml that contains some information related to the dataset such as mean and standard derivation to normalize data later.
2. Then to run the training you can run by setting some parameters:
python main.py --root_path '/data/' \
               --n_classes [number of classes] \
               --is_train \
               --is_rgb \
               --is_scale \
               --n_epochs 100000 \
               --batch_size 64
and for test 
python main.py --root_path '/data/' \
               --n_classes [number of classes] \
               --is_test \
               --is_rgb \
               --is_scale \
               --n_epochs 100000 \
               --batch_size 64               
 
You can check for more parameter options in opts.py file and in [the repository](https://github.com/kenshohara/3D-ResNets-PyTorch).
Check data/results folder for saved checkpoints. 

There is also a Dockerfile that enables one to run everything without installing the necessary tools and libraries.




## 代码使用简介

1. 下载好数据集，
2. 在`train.py`脚本中将`--data-path`设置成`Project_bdaminton_upload/Shot-Transition-Detection-main/MoblieVit/data/badminton/`文件夹绝对路径
3. 下载预训练权重，在`model.py`文件中每个模型都有提供预训练权重的下载地址，根据自己使用的模型下载对应预训练权重
4. 在`train.py`脚本中将`--weights`参数设成下载好的预训练权重路径
5. 设置好数据集的路径`--data-path`以及预训练权重的路径`--weights`就能使用`train.py`脚本开始训练了(训练过程中会自动生成`class_indices.json`文件)
6. 在`predict.py`脚本中导入和训练脚本中同样的模型，并将`model_weight_path`设置成训练好的模型权重路径(默认保存在weights文件夹下)
7. 在`predict.py`脚本中将`img_path`设置成你自己需要预测的图片绝对路径
8. 设置好权重路径`model_weight_path`和预测的图片路径`img_path`就能使用`predict.py`脚本进行预测了
9. 如果要使用自己的数据集，请按照分类数据集的文件结构进行摆放(即一个类别对应一个文件夹)，并且将训练以及预测脚本中的`num_classes`设置成你自己数据的类别数


1. Download the dataset
2. Set `-data-path` to the absolute path of the `fProject_bdaminton_upload/Shot-Transition-Detection-main/MoblieVit/data/badminton/` folder after decompression in the `train.py` script.
3. Download the pre-training weights. Each model in the `model.py` file has a download address for the pre-training weights, so download the corresponding pre-training weights according to the model you are using.
4. Set the `--weights` parameter in the `train.py` script to the path of the downloaded pre-training weights
5. Set the path of the dataset `-data-path` and the path of the pre-trained weights `-weights` to start training with the `train.py` script (the `-class_indices.json` file will be generated automatically during the training process)
6. Import the same model in the `predict.py` script as in the training script, and set the `model_weight_path` to the path of the trained model weights (the default is saved in the weights folder)
7. Set `img_path` in the `predict.py` script to the absolute path of the image you want to predict. In this project, my absolute path is 'to_classfid'
8. Set the weight path `model_weight_path` and the predicted image path `img_path` and you can use the `predict.py` script to make predictions
9. If you want to use your own dataset, please follow the file structure of the classification dataset (i.e. one category for one folder) and set the `num_classes` in the training and prediction scripts to the number of categories in your own data


# Object Detection for Moose using Tensorflow's Object Detector API

## TensorFlow Models

This repository contains a number of different models implemented in [TensorFlow](https://www.tensorflow.org):

The [official models](official) are a collection of example models that use TensorFlow's high-level APIs. They are intended to be well-maintained, tested, and kept up to date with the latest stable TensorFlow API. They should also be reasonably optimized for fast performance while still being easy to read. We especially recommend newer TensorFlow users to start here.

The [research models](https://github.com/tensorflow/models/tree/master/research) are a large collection of models implemented in TensorFlow by researchers. They are not officially supported or available in release branches; it is up to the individual researchers to maintain the models and/or provide support on issues and pull requests.

The [samples folder](samples) contains code snippets and smaller models that demonstrate features of TensorFlow, including code presented in various blog posts.

The [tutorials folder](tutorials) is a collection of models described in the [TensorFlow tutorials](https://www.tensorflow.org/tutorials/).

### Contribution guidelines

If you want to contribute to models, be sure to review the [contribution guidelines](CONTRIBUTING.md).

### License

[Apache License 2.0](LICENSE)


## Steps for doing object detection using your own dataset

### 1. Cloning the repository

Clone the tensorflow's models repository using the following command.  
Create tensorflow/models/research/object_detection/images, tensorflow/models/research/object_detection/images/train, and tensorflow/models/research/object_detection/images/test folders.  

```
https://github.com/Kishaan/Moose-Object-Detector.git
```

### 2. Collecting and creating dataset

Collect enough dataset and put them in the object_detection/images folder.    
Copy 80% of the images inside the object_detection/images/train and the rest 20% inside the object_detection/images/test folders.  
Annotate the images in both the folders using [LabelImg](https://github.com/tzutalin/labelImg). The annotated files will be saved as XML files.  

### 3. Creating the TFRecord files which could be fed into the model

The XML files should be converted into CSV files before converting them into TFRecord files. Create a new folder called "data" inside object_detection folder and use the modified version of [datitran's](https://github.com/datitran/raccoon_dataset) xml_to_csv.py script to do this.  
To convert the csv files into TFRecord files, use the generate_tfrecord.py script. Change the "class_text_to_int" function according to the class/classes of your dataset.  

### 4. Installation and compilation

In order to make the object detection API work, install the following packages  

```
sudo apt-get install protobuf-compiler python-pil python-lxml
sudo pip install jupyter
sudo pip install matplotlib
```

```
# From object_detection folder
protoc ./protos/*.proto --python_out=.
```

```
# From tensorflow/models/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

```
# Installing the object_detection library from the models/research directory 
sudo python3 setup.py install
```

```
# generate the tf records using generate_tfrecord.py for training and test images
python3 generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record

python3 generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record
```

### 5. Setting up the configuration and pretrained models

Download the checkpoint and condig files of mobilenet model (or a model of your choice) using the following commands  

```
wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/samples/configs/ssd_mobilenet_v1_pets.config

wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
```
 
Put the config in object_detection/training directory, and extract the ssd_mobilenet_v1 in the research/object_detection directory.  

Create a file named object_detection.pbtxt and edit the details of your classes. In my case it would look like this

```
item {
  id: 1
  name: 'moose'
}
```

Change all the PATH_TO_BE_CONFIGURED points in the config file and change them according to your files. Modify the batch size, checkpoint name/path, num_classes, num_examples, and label_map_path: "training/object-detection.pbtxt"  

### 6. Training the model with the new dataset

Run the folling command within research/object_detection folder  

```
python3 legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config
```

Train it until the loss gets down below 1.0. It takes around 8 hours to train in a decent GPU machine.  

### 7. Testing the model

Run the export_inference_graph.py in research/object_detection/ using the following command. Replace the arguments pipeline_config, trained_checkpoint, and output_directory according to your files. trained_checkpoit is the .ckpt file in the training folder with the largest step.  

```
python3 export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/ssd_mobilenet_v1_pets.config \
    --trained_checkpoint_prefix training/model.ckpt-10856 \
    --output_directory moose_inference_graph
```

Copy your test images into models/object_detection/test_images directory and run the jupyter notebook named 'object_detection_tutorial.ipynb'.  

Change the Variables section and change the model name, and the paths to the checkpoint and the labels as shown below.  

```
# What model to download.
MODEL_NAME = 'moose_inference_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 1
``` 

Ignore the 'Download model' section and change the test_image_paths in the Detection section.    

```
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.jpg'.format(i)) for i in range(1, 10) ]
```

The above code expects the names of your test images to be in the format: 1.jpg, 2.jpg ...  

Run both the cells below to see how well the model has performed on the test images.     

## References

1. https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9
2. https://pythonprogramming.net/creating-tfrecord-files-tensorflow-object-detection-api-tutorial/
3. https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-2-converting-dataset-to-tfrecord-47f24be9248d
4. Moose Image: https://www.nps.gov/romo/learn/nature/moose.htm
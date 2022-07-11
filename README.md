# Face Recognition using Tensorflow

mkdir models
cd models
download model [20180402-114759] below and add to this folder

## Performance
The accuracy on LFW for the 
model [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) is 0.99650+-0.00252.
 A description of how to run the test can be found on the page [Validate on LFW]
(https://github.com/davidsandberg/facenet/wiki/Validate-on-lfw). 
Note that the input images to the model need to be standardized using fixed image standardization 
(use the option `--use_fixed_image_standardization` when running e.g. `validate_on_lfw.py`).

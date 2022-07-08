python3 src/classifier.py TRAIN ../facenet/datasets/lfw/lfw_mtcnnpy_160 \
    ./models/20180402-114759/20180402-114759.pb \
    ./models/lfw_classifier.pkl \
    --batch_size 1000 \
    --min_nrof_images_per_class 40 \
    --nrof_train_images_per_class 35 \
    --use_split_dataset

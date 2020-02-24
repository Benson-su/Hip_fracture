import shutil
import os
import tensorflow as tf
import glob
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import numpy as np
from imgaug import augmenters as iaa
from configparser import ConfigParser
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Dropout
from models.keras import ModelFactory
from skimage.exposure import equalize_hist, equalize_adapthist
from keras import backend as K
from PIL import Image

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

'''
# set GPU allow growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
'''

def main():
    # parser config
    config_file = "./config_pipe.ini"
    cp = ConfigParser()
    cp.read(config_file)
    output_dir = cp["DEFAULT"].get("output_dir")
    base_model_name = cp["DEFAULT"].get("base_model_name")
    use_base_model_weights = cp["TRAIN_PXR"].getboolean("use_base_model_weights")
    use_trained_model_weights = cp["TRAIN_PXR"].getboolean("use_trained_model_weights")
    train_weights_name = cp["TRAIN"].get("output_weights_name")
    output_weights_name = cp["TRAIN_PXR"].get("output_weights_name")
    train_image_folder = cp["TRAIN_PXR"].get("train_image_folder")
    validation_image_folder = cp["TRAIN_PXR"].get("validation_image_folder")
    test_image_folder = cp["TRAIN_PXR"].get("test_image_folder")
    positive_class_weight = cp["TRAIN_PXR"].getfloat("positive_class_weight", 1.0)
    epochs = cp["TRAIN_PXR"].getint("epochs")
    batch_size = cp["TRAIN_PXR"].getint("batch_size")
    initial_learning_rate = cp["TRAIN_PXR"].getfloat("initial_learning_rate")
    generator_workers = cp["TRAIN_PXR"].getint("generator_workers")
    image_dimension = cp["TRAIN_PXR"].getint("image_dimension")
    train_steps = cp["TRAIN_PXR"].get("train_steps")
    patience_reduce_lr = cp["TRAIN_PXR"].getint("patience_reduce_lr")
    min_lr = cp["TRAIN_PXR"].getfloat("min_lr")
    validation_steps = cp["TRAIN_PXR"].get("validation_steps")
    show_model_summary = cp["TRAIN_PXR"].getboolean("show_model_summary")
    preprocessing_function = cp["TRAIN_PXR"].get("preprocessing_function")
    rotate_angle = cp["TRAIN_PXR"].getint("rotate_angle", 10)
    zoom_range = cp["TRAIN_PXR"].getfloat("zoom_range", 0.1)
    width_shift_range = cp["TRAIN_PXR"].getfloat("width_shift_range", 0)
    height_shift_range = cp["TRAIN_PXR"].getfloat("height_shift_range", 0)
    brightness_range_max= cp["TRAIN_PXR"].getfloat("brightness_range_max",1)
    brightness_range_min= cp["TRAIN_PXR"].getfloat("brightness_range_min",1)
    brightness_range=[brightness_range_min,brightness_range_max]
    horizontal_flip = cp["TRAIN_PXR"].getboolean("horizontal_flip", True)
    vertical_flip = cp["TRAIN_PXR"].getboolean("vertical_flip", True)
    final_dropout = cp["TRAIN_PXR"].getfloat("final_dropout", 0.5)
    image_colors = cp["TRAIN_PXR"].getint("image_colors", 1)
    rescale = cp["TRAIN_PXR"].getfloat("rescale", 1./255)
    if image_colors==1:
        img_colormode="grayscale"
    else:
        img_colormode="rgb"

    # end parser config

    # check output_dir, create it if not exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    running_flag_file = os.path.join(output_dir, ".training_pxr.lock")
    if os.path.isfile(running_flag_file):
        raise RuntimeError("A process is running in this directory!!!")
    else:
        open(running_flag_file, "a").close()
    try:
        print(f"backup config file to {output_dir}")
        shutil.copyfile(config_file, os.path.join(output_dir, os.path.split(config_file)[1]))

        train_counts = len(glob.glob(f"{train_image_folder}/*/*.png"))
        dev_counts = len(glob.glob(f"{validation_image_folder}/*/*.png"))
        print(f"train_counts: {train_counts}, dev_counts: {dev_counts}")

        # compute steps
        if train_steps == "auto":
            train_steps = int(train_counts / batch_size)
        else:
            try:
                train_steps = int(train_steps)
            except ValueError:
                raise ValueError(f"""
                train_steps: {train_steps} is invalid,
                please use 'auto' or integer.
                """)
        print(f"** train_steps: {train_steps} **")

        if validation_steps == "auto":
            validation_steps = int(dev_counts / batch_size)
        else:
            try:
                validation_steps = int(validation_steps)
            except ValueError:
                raise ValueError(f"""
                validation_steps: {validation_steps} is invalid,
                please use 'auto' or integer.
                """)
        print(f"** validation_steps: {validation_steps} **")

        print("** load model **")
        if use_trained_model_weights:
            model_weights_file = os.path.join(output_dir, train_weights_name)
        else:
            model_weights_file = None

        model_factory = ModelFactory()
        model = model_factory.get_model(
            # FIXME (hard coding)
            5,
            model_name=base_model_name,
            use_base_weights=use_base_model_weights,
            weights_path=model_weights_file,
            input_shape=(image_dimension, image_dimension, image_colors))
        fc = Dense(1024, activation="relu")(model.layers[-2].output)
        fc = Dropout(final_dropout)(fc)
        new_output = Dense(2, activation="softmax")(fc)

        new_model = Model(model.input, new_output)

        #new_model = multi_gpu_model(tmp_model, gpus=2)

        if show_model_summary:
            new_model.summary()

        if preprocessing_function == "equalize_hist":
            preprocessing_function = equalize_hist
        # elif preprocessing_function == "equalize_adapthist":
        #     preprocessing_function = equalize_adapthist
        # TODO more preprocessing function options here
        else:
            preprocessing_function = None

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential([
            sometimes(iaa.Fliplr(0.05)), # horizontal flips
            sometimes(iaa.Crop(percent=(0, 0.05))), # random crops
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(0.5,
                iaa.GaussianBlur(sigma=(0.5, 1.5))
                ),
            # Strengthen or weaken the contrast in each image.
            sometimes(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            sometimes(iaa.AdditiveGaussianNoise(scale=(0.1*255), per_channel=0.5)),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            sometimes(iaa.Multiply((0.9, 1.1))),
            
            #jpeg compress
            #tune color
            sometimes(iaa.WithChannels(0, iaa.Add((0, 10)))),
            sometimes(iaa.WithChannels(1, iaa.Add((0, 10)))),
            sometimes(iaa.WithChannels(2, iaa.Add((0, 10)))),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                rotate=(-5, 5),
            #    shear=(-8, 8)
                ))

            ], random_order=True) # apply augmenters in random order
        print("** create image generators **")
        train_generator = ImageDataGenerator(
                rescale=rescale,
                preprocessing_function=seq.augment_image,
        )
        validation_generator = ImageDataGenerator(
                rescale=rescale,
                preprocessing_function=seq.augment_image,
        )
        test_generator = ImageDataGenerator(
                rescale=rescale,
                preprocessing_function=None,
        )

        output_weights_path = os.path.join(output_dir, output_weights_name)
        print(f"** set output weights path to: {output_weights_path} **")
        # callback
        optimizer = Adam(lr=initial_learning_rate)
        new_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"])
        callbacks = [
            ModelCheckpoint(
                output_weights_path,
                save_weights_only=True,
                save_best_only=True,
                verbose=1,
            ),
            TensorBoard(log_dir=os.path.join(output_dir, "tf-log"), batch_size=batch_size),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_reduce_lr,
                              verbose=1, mode="min", min_lr=min_lr),
            CSVLogger(filename=os.path.join(output_dir, "log.csv"), separator=","),
        ]
        print("** start training **")
        new_model.fit_generator(
            generator=train_generator.flow_from_directory(
                train_image_folder,
                target_size=(image_dimension, image_dimension),
                batch_size=batch_size,
                color_mode=img_colormode,
                class_mode="categorical",
                seed=0,
            ),
            #class_weight={0: positive_class_weight, 1: 1.0},
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=validation_generator.flow_from_directory(
                validation_image_folder,
                target_size=(image_dimension, image_dimension),
                batch_size=batch_size,
                color_mode=img_colormode,
                class_mode="categorical",
                seed=0,
            ),
            validation_steps=validation_steps,
            callbacks=callbacks,
            workers=generator_workers,
        )
        print("** start evaluating test set **")
        results = new_model.evaluate_generator(test_generator.flow_from_directory(
            test_image_folder,
            target_size=(image_dimension, image_dimension),
            batch_size=batch_size,
            color_mode=img_colormode,
            class_mode="categorical",
            shuffle=False,
            seed=0,
        ))
        print(results)

        with open(os.path.join(output_dir, "log.csv"), "a") as f:
            f.write("\n")
            f.write(str(results))
    finally:
        os.remove(running_flag_file)
    

if __name__ == "__main__":
    main()

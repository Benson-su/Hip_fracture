import sys
import shutil
import cv2
import glob
import numpy as np
import pandas as pd
from pandas import Series
import os
import time
import tensorflow as tf
from configparser import ConfigParser
from keras import backend as kb
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from models.keras import ModelFactory

#from tensorflow.keras.vis.visualization import visualize_cam, overlay
from vis.utils.utils import apply_modifications
from vis.visualization import visualize_cam, overlay
from keras import activations
from keras import backend as K
from matplotlib import cm
from skimage.exposure import equalize_hist


from PIL import ImageFile
from keras.applications.xception import Xception
from keras.applications import InceptionResNetV2
ImageFile.LOAD_TRUNCATED_IMAGES = True

output_dir_classes=[
#    "Nonaug",
#    "crop",
#    "rotate",
#    "shift_reduce",
#    "zoom",
#    "blur",
#    "bright",
#    "noise",
#    "color_jitter",
#    "contrast",
#    "pipe_shift_reduce"
#    "double_pipe"
    "pipe_reduce_aug_double",
#    "ori_keras"
]

test_classes=[
#    "CGMH",
    "Standford_all"
#    "Standford_test1",
#    "Standford_test2",
#    "JHU",
#    "Boston"
]

class_folders=[
    "hipfx",
    "normal"
]

#def find_max_index(heatmap, image_for_vis):



def get_output_layer(model, layer_name):
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

def gradcam(model, x):
    # 取得影像的分類類別
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    pred_class = np.argmax(preds[0])
#   nt(pred_class)
    #得影像分類名稱
#   d_class_name = imagenet_utils.decode_predictions(preds)[0][0][1]
    
    #測分類的輸出向量
    pred_output = model.output[:, pred_class]
    
   # 後一層 convolution layer 輸出的 feature map
    #sNet 的最後一層 convolution layer
    last_conv_layer = model.get_layer('conv5_block16_2_conv')
    #last_conv_layer = model.get_layer("block14_sepconv2_bn")
    # 求得分類的神經元對於最後一層 convolution layer 的梯度
    grads = K.gradients(pred_output, last_conv_layer.output)[0]
    print(grads) 
    # 求得針對每個 feature map 的梯度加總
    pooled_grads = K.sum(grads)
    
    # K.function() 讓我們可以藉由輸入影像至 `model.input` 得到 `pooled_grads` 與
    # `last_conv_layer[0]` 的輸出值，像似在 Tensorflow 中定義計算圖後使用 feed_dict
    # 的方式。
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    
    # 傳入影像矩陣 x，並得到分類對 feature map 的梯度與最後一層 convolution layer 的 
    # feature map
    pooled_grads_value, conv_layer_output_value = iterate([x])
    
    # 將 feature map 乘以權重，等於該 feature map 中的某些區域對於該分類的重要性
    for i in range(pooled_grads_value.shape[0]):
        conv_layer_output_value[:, :, i] *= (pooled_grads_value[i])
        
    # 計算 feature map 的 channel-wise 加總
    heatmap = np.sum(conv_layer_output_value, axis=-1)
    if pred_class == 0:
        a='normal'     
    else :
        a='fracture' 
    return heatmap, a


def plot_heatmap(heatmap, img_path, a):
    # ReLU
    heatmap = np.maximum(heatmap, 0)
    
    # 正規化
    heatmap /= np.max(heatmap)
    
    # 讀取影像
    img = cv2.imread(img_path)
    
    fig, ax = plt.subplots()
    
    im = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (img.shape[1], img.shape[0]))

    # 拉伸 heatmap
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)
    
    # 以 0.6 透明度繪製原始影像
    ax.imshow(im, alpha=0.6)
    
    # 以 0.4 透明度繪製熱力圖
    ax.imshow(heatmap, cmap='jet', alpha=0.4)
#    plt.figure(figsize=(1.5,1))
#    plt.title(pred_class_name)
#    plt.show()
    plt.savefig('pred_class_{}_{}'.format(a,i), dpi=256)
    print('pltsave')


def create_cam_image(image_for_model, image_for_vis, cam_model, label_ground_truth, label_prediction,
                     output_dir, filename, DumpMode):
    def create_single_class_cam_image(class_index):
        """
        Create CAM for a single class

        :param class_index: int
        :return: numpy.array
        """
        heatmap = visualize_cam(
            model=cam_model,
            layer_idx=-1,
            filter_indices=class_index,
            seed_input=image_for_model,
            # backprop_modifier="relu",
        )
        cmap = cm.jet
        mapper = cm.ScalarMappable(norm=None, cmap=cmap)
        jet_heatmap = np.uint8(mapper.to_rgba(heatmap)[..., :3] * 255)
        return cv2.cvtColor(overlay(jet_heatmap, image_for_vis), cv2.COLOR_RGB2BGR)
        
    images = [image_for_vis]
    for ci in range(2):
        images.append(create_single_class_cam_image(ci))
    
    # write cam image
    if DumpMode=="default":
        output_path = os.path.join(output_dir, f"{time.time()}_{label_ground_truth}_{label_prediction}.png")
    elif DumpMode=="filename":
        output_path = os.path.join(output_dir, f"{filename}_{label_ground_truth}_{label_prediction}.png")
    concatenated_image = np.concatenate(images, axis=1)
    print(f"writing image to {output_path}")
    cv2.imwrite(output_path, concatenated_image)

    return images


def create_cam_image_under_cutoff(image_for_model, image_for_vis, cam_model, label_ground_truth, label_prediction,
                     output_dir, filename, DumpMode):

    images = [image_for_vis]
    #for i in range(2):
    #    images.append(image_for_vis)

    # write cam image
    if DumpMode=="default":
        output_path = os.path.join(output_dir, f"{time.time()}_{label_ground_truth}_{label_prediction}.png")
    elif DumpMode=="filename":
        output_path = os.path.join(output_dir, f"{filename}_{label_ground_truth}_{label_prediction}.png")
    concatenated_image = np.concatenate(images, axis=1)
    print(f"writing image to {output_path}")
    cv2.imwrite(output_path, concatenated_image)

    return images


def get_image_for_vis(image):
    image_for_vis = np.copy(image)
    if np.max(image_for_vis) <= 1:
        image_for_vis *= 255
    image_for_vis = Image.fromarray(np.uint8(image_for_vis))
    image_for_vis = np.array(image_for_vis.convert("RGB"))
    return image_for_vis

def check_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)

def preprocessing(image_folder, output_dir):
    output_path = []#cam subfolder, hipfx/normal...etc

    for class_folder in class_folders:

        old_image_folder = os.path.join(image_folder, class_folder)
        files = os.listdir(old_image_folder)

        tmppath = os.path.join(output_dir, class_folder)# square image folder
        output_path.append(tmppath)
        check_folder(tmppath)
        
        for afile in files:
            path = os.path.join(old_image_folder, afile)
            img = Image.open(path)
            new_image = make_square(img)
            afile = afile[:len(afile)-3]+"png"
            filesave = os.path.join(tmppath, afile)
            new_image.save(filesave)
            print("image saved")
            print(afile)
        
    return output_path

def make_square(im, min_size=512, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size-x)/2), int((size-y)/2)))
    return new_im
                    
                    
def main():
    # config parse
    config_file = "./config_cam.ini"
    cp = ConfigParser()
    cp.read(config_file)

    output_dir = cp["DEFAULT"].get("output_dir")
    base_model_name = cp["DEFAULT"].get("base_model_name")
    train_weights_name = cp["TRAIN_PXR"].get("output_weights_name")
    image_dimension = cp["TRAIN_PXR"].getint("image_dimension")
    preprocessing_function = cp["TRAIN_PXR"].get("preprocessing_function")
    output_folder_name = cp["CAM"].get("output_folder_name")
    cutoff = cp["CAM"].getfloat("cutoff")
    image_colors = cp["TRAIN_PXR"].getint("image_colors", 1)
    rescale = cp["TRAIN_PXR"].getfloat("rescale", 1./255)
    batch_size = cp["TRAIN_PXR"].getint("batch_size", 1)
    test_steps = cp["CAM"].get("test_steps")
    shuffle = cp["CAM"].get("shuffle")
    DumpMode = cp["CAM"].get("DumpMode")
    zeropad_dir = cp["CAM"].get("zeropad_folder")


    if image_colors==1:
        img_colormode="grayscale"
    else:
        img_colormode="rgb"

    try:
        image_folder = "./"+str(sys.argv[1])
    except:
        image_folder = cp["CAM"].get("image_folder")

    if shuffle=="False":
        shuffle=False
    else:
        shuffle=True

    # define new data input for square padding 
    '''
    output_path = preprocessing(image_folder, output_dir)
    weight_dir = output_dir
    check_folder(zeropad_dir)

    
    for path in output_path:
        shutil.move(path, zeropad_dir)
    
    image_folder = zeropad_dir
    '''
    # model

    for output_dir_class in output_dir_classes: 
        for test_class in test_classes:

            output_dir = output_dir[:(output_dir.rfind("/")+1)] + output_dir_class
            
            image_folder = image_folder[:(image_folder.rfind("/") + 1)] + test_class

            print("!!!!!!!!!!!!image_foder: !!!!!!!!!!!!!!")
            print(image_folder)
            print("!!!!!!!!!!!!output_dir: !!!!!!!!!!!!!!!!")
            print(output_dir)
            #model = InceptionResNetV2(input_shape=(299, 299, 3),weights='imagenet')
            #if use_trained_model_weights:
            model_weights_file = os.path.join(output_dir, train_weights_name)
            #else:
            #    model_weights_file = None
            model_factory = ModelFactory()
            model = model_factory.get_model(
                5,
                model_name=base_model_name,
                use_base_weights=False,
                weights_path=None,
                input_shape=(image_dimension, image_dimension, image_colors))
            fc = Dense(1024, activation="relu")(model.layers[-2].output)
            fc = Dropout(0.5)(fc)
            new_output = Dense(2, activation="softmax")(fc)
            cam_model = Model(model.input, new_output)
            print("load trained model weights")
            print("image_folder: "+image_folder)
            cam_model.load_weights(model_weights_file)

            # replace sigmoid to linear
            '''
            cam_model.layers[-1].activation = activations.linear
            cam_model = apply_modifications(cam_model)
            '''
            # define prediction function for cam
            final_conv_layer = get_output_layer(cam_model, model_factory.get_last_conv_layer(base_model_name))

            #final_conv_layer = get_output_layer(cam_model,"conv_7b_ac")
            get_predictions = kb.function(
                [cam_model.layers[0].input],
                [cam_model.output, final_conv_layer.output]
            )

            # FIXME: load from util package
            if preprocessing_function == "equalize_hist":
                preprocessing_function = equalize_hist
            else:
                preprocessing_function = None

            print(f"preprocessing_function: {preprocessing_function}")

            # generator
            generator_factory = ImageDataGenerator(
                rescale=rescale,
                preprocessing_function=preprocessing_function,
            )

            generator = generator_factory.flow_from_directory(
                directory=image_folder,
                target_size=(image_dimension, image_dimension),
                batch_size=1,
                color_mode=img_colormode,
                class_mode="categorical",
                shuffle=shuffle,
                seed=0,
            )

            # cam output dir
            cam_output_dir = os.path.join(output_dir, test_class)
            
            # label: [hip, normal]
            image_count = len(glob.glob(f"{image_folder}/*/*"))
            print(f"image_count: {image_count}")
            if test_steps == "auto":
                test_steps = int(image_count / batch_size)
            else:
                try: 
                    test_steps = int(test_steps)
                except ValueError:
                    raise ValueError(f"""test_steps: {test_steps} is invalid, please use 'auto' or integer.""")

            if not os.path.isdir(cam_output_dir):
                os.makedirs(cam_output_dir)

            # image_count = 1
            # shuffle must be False to insure file order
            filename = generator.filenames

            generator.reset()

            counter = 0
            for i in range(image_count):
                x_batch, y_batch = next(generator)
                predictions = get_predictions([x_batch])
                onefilename = filename[i]
                onefilename = onefilename[(onefilename.find("/")+1):]
                onefilename = onefilename[:onefilename.find(".")]
            #    print("filename[i]: "+str(onefilename))
            #    if onefilename.find("_") != -1:
            #        continue
            #    print("filename: "+str(onefilename))
                # convert the original image to RGB
                image_for_model = x_batch[0]
                image_for_vis = get_image_for_vis(image_for_model)
                if  max(predictions[0][0]) > cutoff:
                    predict_accu = int(np.argmax(predictions[0][0]))
                else:
                    predict_accu = 1
                create_cam_image(
                    image_for_model=image_for_model,
                    image_for_vis=image_for_vis,
                    cam_model=cam_model,
                    label_ground_truth=int(np.argmax(y_batch[0])),
                    label_prediction=predict_accu,
                    output_dir=cam_output_dir,
                    filename=onefilename,
                    DumpMode=DumpMode
                )
                
                # write image with probability to csv

                csv_output = "./prob_standford/fx_prob_" + test_class + "_"+ output_dir_class + ".csv"

                fp = open(csv_output, "a")

                fp.write(str(onefilename).ljust(25)+"   "+str(int(np.argmax(y_batch[0]))).ljust(20)+"    "+str(float(predictions[0][0][0])).ljust(20)+"\n")
                fp.close()

if __name__ == "__main__":
    main()

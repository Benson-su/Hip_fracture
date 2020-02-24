import os
from random import shuffle, seed


def check_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)

seed(0)
root = "./PXRV2_mixed"
class_folders = [
        "hipfx",
        "normal"
]

for class_folder in class_folders:
    source_folder = os.path.join(root, class_folder)
    print("source_folder: "+str(source_folder))
    train_folder = f"./train_pxr_80_V2_crop/{class_folder}"
    dev_folder = f"./dev_pxr_80_V2_crop/{class_folder}"
    test_folder = f"./test_pxr_80_V2_crop/{class_folder}"
    check_folder(train_folder)
    check_folder(dev_folder)
    check_folder(test_folder)
    image_files = os.listdir(source_folder)
    shuffle(image_files)
    print("image count"+str(len(image_files)))
    train_count = int(0.8 * len(image_files))
    dev_count = int(0.1 * len(image_files))
    for image_file in image_files[:train_count]:
        os.symlink(
            os.path.join("../../",source_folder, image_file),
            os.path.join(train_folder, image_file),
        )
    for image_file in image_files[train_count:(train_count+dev_count)]:
        os.symlink(
            os.path.join("../../",source_folder, image_file),
            os.path.join(dev_folder, image_file),
        )
    for image_file in image_files[(train_count+dev_count):]:
        os.symlink(
            os.path.join("../../",source_folder, image_file),
            os.path.join(test_folder, image_file),
        )


import os
import shutil
import numpy as np
import pandas as pd
from random import shuffle, seed


def check_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)

root = "./oridata/ori"

PXRdata = pd.read_csv("PXRV1_result.csv")
source_folder = os.path.join(root)
image_files = os.listdir(source_folder)

root_file = "./oridata/ori/"
target_hipfx = "./oridata/hipfx/"
target_normal = "./oridata/normal/"
for image_file in image_files:
	tmp_file = PXRdata.loc[PXRdata['imgname']==image_file]
	source_file = root_file + image_file
	
	if (tmp_file['hipfx']==1).bool() is True:
#		print(image_file)
		shutil.move(source_file, target_hipfx)
	else:
		shutil.move(source_file, target_normal)

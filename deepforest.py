import os
from deepforest import get_data
from deepforest import deepforest
from deepforest import utilities
from deepforest import preprocess

#Se abre el archivo que genera la etiquetada de las palmas
#Para etiquetar los arboles que se quieren identificar es necesario hacerlo en un programa para etiquetado que se llama LabelImg
cast_xml = get_data("C:/Users/labels_entrenamiento.xml")
annotation = utilities.xml_to_annotations(cast_xml)
annotation.head()

#Se convierte csv
annotation.to_csv("C:/Users/labels_entrenamiento.csv", index=False)

#El tif con el cual se está entrenando el modelo
tif_train = get_data('C:/Users/arboles.tif')

#Se pone una dirección para que se guarden los cortes.
#Patch Size=numero de pixeles para cada corte
crop_dir = 'C:/Users/DeepForest'
train_annotations= preprocess.split_raster(path_to_raster=tif_train,
                                 annotations_file="C:/Users/labels_entrenamiento.csv",
                                 base_dir=crop_dir,
                                 patch_size=1200,
                                 patch_overlap=0.05)

annotations_file= os.path.join(crop_dir, "train_example.csv")
train_annotations.to_csv(annotations_file,index=False, header=None)

test_model=deepforest.deepforest()
test_model.use_release()

#Se le pueden cambiar al modelo diferentes parametros para que encuentre mejor las palmas (https://deepforest.readthedocs.io/en/latest/training.html#config-file)
test_model.config["epochs"] = 5
test_model.config["save-snapshot"] = False

#Se entrena el modelo
test_model.train(annotations=annotations_file, input_type="fit_generator")

test_model.model.save("C:/Users/modelo.h5")
#para usarlo: reloaded = deepforest.deepforest(saved_model="C:/Users/modelo.h5")
#https://deepforest.readthedocs.io/en/latest/getting_started.html#loading-saved-models-for-prediction

#Se calcula la presición del modelo
performance = test_model.evaluate_generator(annotations=annotations_file)
print("Mean Average Precision is: {:.3f}".format(performance))

#Eso es todo de entrenamiento del modelo

import pandas as pd
from deepforest import deepforest
from deepforest import get_data
from matplotlib import pyplot as plt
import os
from deepforest import utilities
from deepforest import preprocess
import rasterio
import numpy as np
from io import BytesIO
from PIL import Image, ImageFile

#Se abre la imagen a la que se le quiere contar las palmas (diferente de la de entrenamiento)
raster_path = 'C:/Users/arboles_diferentes.tif'
raster = Image.open(raster_path)
numpy_image = np.array(raster)
print(numpy_image.shape)

numpy_image=numpy_image[:,:,:3]
plt.imshow(numpy_image)

trained_model_tile = test_model.predict_tile(raster_path,return_plot=True,patch_size=500,patch_overlap=0.15,iou_threshold=0.15)

fig = plt.figure(figsize=(100,100))
plt.imshow(trained_model_tile)
plt.savefig("C:/Users/deep_forest3.png")


#doneeeee



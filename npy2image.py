import numpy as np
from skimage import io
from keras.preprocessing.image import ImageDataGenerator
save_path = '/home/pzp/img_aug/'
npy_image_path = '/home/pzp/PycharmProjects/tianchi/data/tianchi_trainset/trainimages.npy'
npy_mask_path = '/home/pzp/PycharmProjects/tianchi/data/tianchi_trainset/trainmasks.npy'

img_npy = np.load(npy_image_path)
mk_npy = np.load(npy_mask_path)
zero = np.zeros((512, 512 ,3))
datagen = ImageDataGenerator(rotation_range=8,
                             horizontal_flip=True,
                             vertical_flip= True,
                             zoom_range=0.1,
                             shear_range=0.08,
                             height_shift_range=0.05,
                             width_shift_range=0.05,
                             fill_mode='nearest')
def doaug(img, datagen, save_dir,save_prefix,batch_size=1, save_format='png', imgnum = 100):
    i=1
    for batch in datagen.flow(img,
                              batch_size=batch_size,
                              save_to_dir = save_dir,
                              save_prefix = save_prefix,
                              save_format = save_format):
        i+= 1
        if i > imgnum:
            break
for i in range(img_npy.shape[0]):
    img = img_npy[i,:,:,:]
    img = np.transpose(img, (2,1,0))
    mk = mk_npy[i,:,:,:]
    mk = np.transpose(mk, (2, 1, 0))
    zero[:,:,0] = img[:,:,0]
    zero[:,:,2] = mk[:,:,0]
    image = np.reshape(zero,(1,)+zero.shape)
    doaug(image,datagen,save_path,str(i))


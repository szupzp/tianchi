import numpy as np
from skimage import io,transform
import os
import random
import LUNA_train_unet
batch_size = 1
epoch_num = 3
imgs_dir = '/home/pzp/img_aug/'
imgs_list = os.listdir(imgs_dir)
# imgs_list = random.shuffle(imgs_list)
model = LUNA_train_unet.get_unet()
npy_dir = '/home/pzp/tianchi_train_npy/'

imgs_npy = np.ndarray((batch_size,512,512,1),dtype=np.uint8)
masks_npy = np.ndarray((batch_size,64,64,1),dtype=np.uint8)
i = 0
for epoch in range(epoch_num):
    j = 0

    for j in range(len(imgs_list)):
        img = io.imread(imgs_dir + imgs_list[j])

        if img.shape != (512,512,3):
            continue
        final_img = img[:,:,0]
        final_mask = img[:,:,2]
        final_img = final_img/255
        final_mask = final_mask / np.max(final_mask)
        final_mask = final_mask.astype(np.uint8)
        final_mask = transform.resize(final_mask,(64,64),mode='constant',preserve_range = True)
        imgs_npy[i,:,:,0] = final_img
        masks_npy[i,:,:,0] = final_mask
        i+=1
        if i == batch_size:
            # print ('fit model on batch '+str(j/(batch_size)))
            loss = model.train_on_batch(imgs_npy, masks_npy)

            # np.save(os.path.join(npy_dir,'final_train_image','image_%06d.npy'%j),imgs_npy)
            # np.save(os.path.join(npy_dir,'final_train_mask','mask_%06d.npy'%j),masks_npy)
            i = 0
        if(j%100==0):
            print('epoch=%d  ' % epoch + 'step=%d   loss = %08f, dice_coef = %.08f' % (j, loss[0] / 1000, loss[1] / 1000))
        if (j/(batch_size))%50000==0:
            model.save('./model/aug_unet_bigloss_step%d.h5'%int(j/(batch_size)))
            print('save model accessfully')
print('train is compile')
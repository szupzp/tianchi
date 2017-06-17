import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

img = np.load('/home/pzp/result/masks_0150_0296.npy')

itk_img = sitk.ReadImage('/home/pzp/天池AI大赛/比赛数据/train/LKDS-00295.mhd')
img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
print(img_array.shape)
plt.imshow(img_array[100],'gray')
plt.show()
plt.imshow(img[0],'gray')
plt.show()
# showfig, ax = plt.subplots(1, 2, figsize=[8, 8])
# ax[0, 0].imshow(img[0], cmap='gray')
# ax[0, 1].imshow(img_array[90], cmap='gray')
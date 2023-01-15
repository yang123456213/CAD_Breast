from os.path import splitext
import numpy as np
import torch

import segmentation_models_pytorch as smp
# from PIL import Image
# # import pydicom
# import nibabel as nib
import cv2
# from albumentations import Resize
# def resize(img, h, w):
#     fun = Resize(h, w, always_apply=True)(image=img)
#     img = fun['image']
#     return img

# def load(filename):
#     """
#         :return numpy_array(h, w, c)
#     """
#     ext = splitext(filename)[1]
#     if ext in ['.npz', '.npy']:
#         return np.load(filename)
#     elif ext in ['.pt', '.pth']:
#         return torch.load(filename).numpy()
#     elif ext in ['.dcm']:
#         ds = pydicom.dcmread(filename)
#         return ds.pixel_array
#     elif ext in ['.nii', '.gz']:
#         return nib.load(filename).get_fdata()
#     else:
#         return np.asarray(Image.open(filename))


def predict(img, model_dir):
    model = smp.Unet(
        in_channels=3,
        encoder_name='resnet18',  # 'resnet34',
        encoder_weights=None,
        classes=1,
        activation=None,
    )
    device = torch.device('cpu')
    model.to(device=device)
    model.load_state_dict(torch.load(model_dir, map_location=device))


    img = img / 255.  # h, w, c
    # A = cv2.resize(img, (256, 384))
    # resize_img = torch.as_tensor(A).to(device=device, dtype=torch.float32)
    # resize_img = torch.transpose(resize_img, 2, 0)
    # resize_img = torch.transpose(resize_img, 2, 1)
    # 1 3 384 256
    resize_img = cv2.resize(img, (384, 256), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)[np.newaxis, :]  # h, w, c -> 1, c, h, w
    resize_img = torch.as_tensor(resize_img).to(device=device, dtype=torch.float32)
    mask = model.predict(resize_img)  # 模型预测
    mask = np.asarray(mask.squeeze().detach().cpu())  # 转成array
    # mask = resize(mask, img.shape[0], img.shape[1])  # 恢复原图大小
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)  # 恢复原图大小
    img_mask = mask > 0
    mask[img_mask == 0] = 0
    mask[img_mask == 1] = 1
    return mask



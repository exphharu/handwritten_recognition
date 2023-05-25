import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate

"""
画像 : ndarray

resize_affine : 画像をratioの倍率で拡大縮小する
reduction_padding : 93%縮小 + 1padding
reduction_padding : 4パターンの縮小とpaddingがランダムで起こる
random_rotation : 画像を回転させる。画像サイズが大きくなるので超えたらrandom_cropで28x28に戻す
random_crop : 画像を28x28で切り出す
random_erasing : その名の通り。実装したもののあまり効果なし

rotation_crop : rotとcropの合わせたもの
rotation_crop_erasing : rot,crop,rand_erasを合わせたもの

redpad_rotation_crop : reduction_padding, rot, cropを合わせたもの

"""

#affine変換によるリサイズ用def文
def resize_affine(image,ratio):
    #引数に倍率とイメージ（ndarray）を渡す
    img = image
    #print(img.shape)
    img = img.transpose(1,2,0)
    #print(img.shape)
    h, w = img.shape[:2]
    #print('input size : {0}x{1}'.format(h, w))

    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src * ratio
    affine = cv2.getAffineTransform(src, dest)
    #print('output size : {0}x{1}'.format(int(float(w)*ratio), int(float(h)*ratio)))

    
    img2 = cv2.warpAffine(img, affine, (int(float(w)*ratio), int(float(h)*ratio)), cv2.INTER_LANCZOS4)
    #print(img2.shape)

    # 1次元目が消えちゃうので追加して元のshape (1, h, w)に戻す
    img3 = img2[np.newaxis, :, :]
    #print(img3.shape)
    return img3


# 縮小　and Padding  (93%固定)
def reduction_padding(img):
    img2 = resize_affine(img, 0.93)
    #print(img2[0].shape)
    img3 = np.pad(img2[0], 1, "maximum")
    #plt.imshow(img3, cmap='gray')
    #消えた一次元目を追加
    img4 = img3[np.newaxis, :, :]

    return img4

# 縮小　and Padding  (93%, 86%, 79%, 72%) 
def reduction_padding_rand(img):
    red_scale, pad = 0, 0
    a = np.random.randint(1,5)
    #print(a)
    if a == 1: red_scale, pad = 0.93, 1
    if a == 2: red_scale, pad = 0.86, 2
    if a == 3: red_scale, pad = 0.79, 3
    if a == 4: red_scale, pad = 0.72, 4

    img2 = resize_affine(img, red_scale)
    #print(img2[0].shape)
    img3 = np.pad(img2[0], pad, "maximum")
    #plt.imshow(img3, cmap='gray')
    #消えた一次元目を追加
    img4 = img3[np.newaxis, :, :]

    return img4



#画像を回転させる
def random_rotation(image, angle_range):
    _, h, w = image.shape
    #print(image.shape, h, w)
    angle = np.random.randint(*angle_range)
    #print(angle)

    image = np.squeeze(image)
    #print(image.shape)

    image = rotate(image, angle)
    #print(image.shape)
    
    # 1次元目が消えちゃうので追加して元のshape (1, h, w)に戻す
    image = image[np.newaxis, :, :]

    return image

#回転すると画像サイズが変わるので28x28で切り出し箇所をランダムに決める
def random_crop(image, crop_size=(28, 28)):
    _, h, w = image.shape
    #print(h, w)

    # 切り出し箇所はランダム
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])

    # bottomとrightを決める
    bottom = top + crop_size[0]
    right = left + crop_size[1]

    # 決めたtop, bottom, left, rightを使って画像を抜き出す
    image = image[:, top:bottom, left:right]
    return image


def random_erasing(image, p=0.5, s=(0.03, 0.2), r=(0.3, 3)):
    # マスクするかしないか
    if np.random.rand() > p:
        return image

    image2 = np.copy(image)

    # マスクする画素値をランダムで決める
    mask_value = np.random.rand()

    _, h, w = image2.shape
    # マスクのサイズを元画像のs(0.02~0.4)倍の範囲からランダムに決める
    mask_area = np.random.randint(h * w * s[0], h * w * s[1])

    # マスクのアスペクト比をr(0.3~3)の範囲からランダムに決める
    mask_aspect_ratio = np.random.rand() * r[1] + r[0]

    # マスクのサイズとアスペクト比からマスクの高さと幅を決める
    # 算出した高さと幅(のどちらか)が元画像より大きくなることがあるので修正する
    mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
    if mask_height > h - 1:
        mask_height = h - 1
    mask_width = int(mask_aspect_ratio * mask_height)
    if mask_width > w - 1:
        mask_width = w - 1

    top = np.random.randint(0, h - mask_height)
    left = np.random.randint(0, w - mask_width)
    bottom = top + mask_height
    right = left + mask_width

    #print(mask_value)
    #print(top, bottom, left, right)
    #print(image2.shape, type(image2))
    image2[0, top:bottom, left:right].fill(mask_value)

    #image2[0, 10:20, 10:20].fill(mask_value)
    return image2


def rotation_crop(images, angle_range=(-15, 15)):
    """
    input : images (num, 1, 28, 28)
    一枚ずつ処理する
    """
    num = len(images)
    #print(num)
    
    images2 = np.empty((num,1,28,28)) #出力格納用

    for i in range(num):
        img = np.copy(images[i])

        img2 = random_rotation(img, angle_range)
        if img2.shape[1] != 28:
            img3 = random_crop(img2)
        else:
            img3 = img2    

        img4 = np.where(img3 == 0, 1, img3)

        #クッキリモード
        #img4 = np.where(img4 > 0.25 , 1, img4)

        images2[i] = img4

    return images2    

    
def rotation_crop_erasing(images, angle_range=(-15, 15)):
    """
    input : images (num, 1, 28, 28)
    一枚ずつ処理する
    """
    num = len(images)
    #print(num)
    
    images2 = np.empty((num,1,28,28)) #出力格納用

    for i in range(num):
        img = np.copy(images[i])

        img2 = random_rotation(img, angle_range)
        if img2.shape[1] != 28:
            img3 = random_crop(img2)
            img4 = random_erasing(img3)
        else:
            img4 = img2    

        img5 = np.where(img4 == 0, 1, img4)

        #クッキリモード
        #img4 = np.where(img4 > 0.25 , 1, img4)

        images2[i] = img5

    return images2

def redpad_rotation_crop(images, angle_range=(-15, 15)):

    num = len(images)
    #print(num)
    
    images2 = np.empty((num,1,28,28)) #出力格納用

    # 一枚ずつ処理
    for i in range(num):
        img = np.copy(images[i])

        img2 = reduction_padding_rand(img)
        img3 = random_rotation(img2, angle_range)

        #回転した場合は画像サイズが変わるのでcropする
        if img3.shape[1] != 28:
            img4 = random_crop(img3)
        else:
            img4 = img3    

        img5 = np.where(img4 == 0, 1, img4)

        #クッキリモード
        #img4 = np.where(img4 > 0.25 , 1, img4)

        images2[i] = img5

    return images2    
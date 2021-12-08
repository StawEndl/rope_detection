import PIL.Image as Image
import os, cv2, random
import numpy as np
import glob
import xml.etree.ElementTree as ET
from skimage import exposure, img_as_float

def calculate_IoU(da1, da2):
    """
    computing the IoU of two boxes.
    Args:
        box: (xmin, ymin, xmax, ymax),通过左下和右上两个顶点坐标来确定矩形位置
    Return:
        IoU: IoU of box1 and box2.
    """
    pxmin, pymin, pxmax, pymax, _ = da1
    gxmin, gymin, gxmax, gymax = da2

    parea = (pxmax - pxmin) * (pymax - pymin)  # 计算P的面积
    garea = (gxmax - gxmin) * (gymax - gymin)  # 计算G的面积

    # 求相交矩形的左下和右上顶点坐标(xmin, ymin, xmax, ymax)
    xmin = max(pxmin, gxmin)  # 得到左下顶点的横坐标
    ymin = max(pymin, gymin)  # 得到左下顶点的纵坐标
    xmax = min(pxmax, gxmax)  # 得到右上顶点的横坐标
    ymax = min(pymax, gymax)  # 得到右上顶点的纵坐标

    # 计算相交矩形的面积
    w = xmax - xmin
    h = ymax - ymin
    if w <=0 or h <= 0:
        return 0

    area = w * h  # G∩P的面积
    # area = max(0, xmax - xmin) * max(0, ymax - ymin)  # 可以用一行代码算出来相交矩形的面积
    # print("G∩P的面积是：{}".format(area))

    # 并集的面积 = 两个矩形面积 - 交集面积
    IoU = area / (parea + garea - area)

    return IoU


def is_inter(xys_small, da2):
    # print(len(xys_small))
    for xy_small in xys_small:
        IoU = calculate_IoU( xy_small, da2)
        # print(IoU)
        if IoU != 0:
            return 1
    return 0

def get_xys_labels_small(path, name, change_size, x1, y1):
    data = []
    # -------------------------------------------------------------#
    #   对于每一个xml都寻找box
    # -------------------------------------------------------------#
    if not os.path.exists(path+name):
        txt = open("D:/paCong/person_dog/labels/"+name[:-3]+"txt", "r")
        da = txt.readline()
        while da:
            d = da.split(" ")
            label, x, y, w, h = d[0], d[1], d[2], d[3], d[4]


    path = path+name
    tree = ET.parse(path)
    # height = int(tree.findtext('./size/height'))
    # width = int(tree.findtext('./size/width'))
    # if height <= 0 or width <= 0:
    #     continue

    # -------------------------------------------------------------#
    #   对于每一个目标都获得它的宽高
    # -------------------------------------------------------------#
    for obj in tree.iter('object'):
        label = 0
        if obj.findtext('name') == "rope":
            label = 1
        xmin = int(int(obj.findtext('bndbox/xmin')) * change_size) + x1
        ymin = int(int(obj.findtext('bndbox/ymin')) * change_size) + y1
        xmax = int(int(obj.findtext('bndbox/xmax')) * change_size) + x1
        ymax = int(int(obj.findtext('bndbox/ymax')) * change_size) + y1


        data.append([xmin, ymin, xmax, ymax, label])

    return data
path_big = "D:/paCong/background_new/fengjing/"
fileNames_big = os.listdir(path_big)

path_sma = "D:/paCong/rope_pet_labeld/"
fileNames_sma = os.listdir(path_sma)

path_sma_label = "D:/paCong/label/"
# path_sma_mask = "D:/paCong/mask/"

save_path = "D:/paCong/big_small/"
# save_path_mask = "D:/paCong/mask_big/"
max_num_small = 8


random_index_big_imgs = []
random_index_big_img = True
val = random_index_big_img
len_data=500 if val else len(fileNames_big)

txt_path = save_path+"2007_"+"val.txt" if val else save_path+"2007_"+"train.txt"
txt_file = open(txt_path, "w")
for index_big_file_name in range(len_data):#len(fileNames_big)
    while random_index_big_img:
        index_big_file_name = random.randint(0, len(fileNames_big)-1)
        if index_big_file_name not in random_index_big_imgs:
            random_index_big_imgs.append(index_big_file_name)
            break
    # print("\nindex_big_file_name", index_big_file_name)
    print("\nindex_big_file_name", index_big_file_name if not random_index_big_img else len(random_index_big_imgs))
    img_big = ""
    try:
        img_big = np.array(Image.open(path_big + fileNames_big[index_big_file_name]))
    except:
        continue
    img_big = cv2.cvtColor(img_big, cv2.COLOR_RGB2BGR)
    h_big, w_big, _ = img_big.shape

    xys_small = []
    num_small = random.randint(1, max_num_small)

    xys_labels_small = []
    got_num = 0
    for num in range(num_small):
        print("num", num, end="    ")
        # print("max_num")
        # while True:
        #     print("while")
            # y1 = random.randint(0, h_big)
            # x1 = random.randint(0, w_big)

        ind_small_photo = random.randint(0, len(fileNames_sma)-1)
        if not os.path.exists(path_sma_label + fileNames_sma[ind_small_photo][:fileNames_sma[ind_small_photo].find(".")] + ".xml"):
            continue
        img_sma = Image.open(path_sma + fileNames_sma[ind_small_photo])



        # wrong = 0
        # tree = ET.parse(path_sma_label + fileNames_sma[ind_small_photo][:fileNames_sma[ind_small_photo].find(".")] + ".xml")
        # for obj in tree.iter('object'):
        #     if obj.findtext('name') == "rope" and (not os.path.exists(path_sma_mask + fileNames_sma[ind_small_photo].replace(".jpg", ".png"))):
        #         wrong = 1
        #         break
        # if wrong==1:
        #     continue

        # print(img_sma.size)
        time = 0
        change_size = 1
        print("time:   ", end="")
        while True:
            print(time, end="    ")
            if time > 20:
                break
            time += 1
            # print("while")
            # change_size = random.randint(25, 400) / 100#增大或者缩小
            change_size = random.randint(25, 40) / 100  # 缩小
            new_size = (int(img_sma.size[0] * change_size), int(img_sma.size[1] * change_size))
            # img_sma = img_sma.resize((int(img_sma.size[0]*change_size), int(img_sma.size[1]*change_size)))
            # img_sma = np.array(img_sma)

            # print((int(img_sma.shape[1]/2), int(img_sma.shape[0]/2)))
            # img_sma.resize((int(img_sma.shape[1]/2), int(img_sma.shape[0]/2), 3))


            # print(w_big , img_sma.shape[1])
            max_y = h_big - new_size[1]
            max_x = w_big - new_size[0]
            if max_x<=0 or max_y<=0:
                continue
            y1 = random.randint(0, max_y-1)
            x1 = random.randint(0, max_x-1)


            x2 = x1 + new_size[0]
            y2 = y1 + new_size[1]

            if x2 <= w_big and y2 <= h_big:
                img_sma = img_sma.resize(new_size)
                img_sma = np.array(img_sma)
                break


        if time > 20:
            continue

        if is_inter(xys_small, (x1, y1, x2, y2)) == 1:
            continue
        img_sma = cv2.cvtColor(img_sma, cv2.COLOR_RGB2BGR)

        light_change = random.randint(1, 37) * 0.1
        # img_sma = exposure.adjust_gamma(img_sma, light_change)

        # # print(img_sma.shape)
        # mask_big = np.zeros((1080, 1920, 3))
        # # print("exit", os.path.exists(path_sma_mask + fileNames_sma[ind_small_photo].replace(".jpg", ".png")), path_sma_mask + fileNames_sma[ind_small_photo].replace(".jpg", ".png"))
        # if os.path.exists(path_sma_mask + fileNames_sma[ind_small_photo].replace(".jpg", ".png")):
        #     mask_sma = cv2.imread(path_sma_mask + fileNames_sma[ind_small_photo].replace(".jpg", ".png"))
        #     # mask_sma = mask_sma.resize(new_size)
        #     # mask_sma = np.array(mask_sma)
        #     # print(mask_sma.shape, img_sma.shape, mask_big[y1: y2, x1: x2].shape)
        #     mask_sma = cv2.resize(mask_sma, (img_sma.shape[1], img_sma.shape[0]))
        #     # print(mask_sma.shape)
        #     # cv2.imshow("sa", mask_sma)
        #
        #     # mask_sma = np.expand_dims(mask_sma, 2)
        #     # print("!", mask_sma.shape)
        # else:
        #     mask_sma = np.zeros((img_sma.shape[0], img_sma.shape[1], 3))
        # # print(img_big.shape, img_big[y1:y2, x1:x2].shape, "img_big[y1:y2, x1:x2]", mask_big[y1:y2, x1:x2].shape,"all" ,mask_sma.shape)
        # mask_big[y1: y2, x1: x2] = mask_sma


        # cv2.imshow("mask_sma", cv2.resize(mask_sma, (600, 600)))
        # cv2.imshow("img_sma", cv2.resize(img_sma, (600, 600)))
        # cv2.waitKey(1000) == ord("n")
        img_big[y1:y2, x1:x2] = img_sma
        # label = int(fileNames_sma[ind_small_photo].split("#")[1])
        label = 2
        xys_small.append((x1, y1, x2, y2, label))

        xys_labels_small.extend(get_xys_labels_small(path_sma_label, fileNames_sma[ind_small_photo][:fileNames_sma[ind_small_photo].find(".")] + ".xml", change_size, x1, y1))
        got_num += 1
        # print(xys_labels_small)
        # for (x1, y1, x2, y2, label) in xys_labels_small:
        #     color = (255, 255, 0)
        #     if label == 0:
        #         color = (0, 0, 255)
        #     cv2.rectangle(img_big, (x1, y1), (x2, y2), color, 2, 2)

    # print(xys_small)
    line = ""
    if val:
        line = path_big +"val_"+ fileNames_big[index_big_file_name]
    else:
        line = path_big + fileNames_big[index_big_file_name]
    for (x1, y1, x2, y2, label) in xys_labels_small:
        line += " " + str(x1) + "," + str(y1) + "," + str(x2) + "," + str(y2) + "," + str(label)
    line += "\n"
    txt_file.write(line)
    if val:
        cv2.imwrite(save_path +"val_"+ fileNames_big[index_big_file_name], img_big)
    else:
        cv2.imwrite(save_path + fileNames_big[index_big_file_name], img_big)

    # cv2.imwrite(save_path_mask +"val_"+ fileNames_big[index_big_file_name], mask_big)
    # cv2.imshow("cat", cv2.resize(img_big, (416, 416)))
    # cv2.waitKey()
    # break


    # IoU = calculate_IoU((1, -1, 3, 1), (0, 0, 2, 2))

txt_file.close()
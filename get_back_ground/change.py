import os
import xml.etree.ElementTree as ET
import PIL.Image as Image
import numpy as np
import cv2

def change_big_txt2yolov5_need_txt():
    val = True
    if val:
        text_file = open("2007_val.txt", "r")
        text_file_2017 = open("val2017.txt", "w")
    else:
        text_file = open("2007_train.txt", "r")
        text_file_2017 = open("train2017.txt", "w")
    count = 0
    while True:
        line = text_file.readline()
        if not line:
            break
        print(count)
        count += 1
        data = line.split(" ")
        text_file_2017.write(data[0] + "\n")

        name = data[0].split("/")[-1]
        label_file = open("./labels/" + name[:name.find(".")] + ".txt", "w")
        for index_data in range(1, len(data)):
            if index_data == len(data)-1:
                da = data[index_data][:-1]
            else:
                da = data[index_data]
            xy_label = da.split(",")
            w = float(int(xy_label[2]) - int(xy_label[0]))
            h = float(int(xy_label[3]) - int(xy_label[1]))
            x1 = (float(xy_label[0]) + w / 2) / 1920
            y1 = (float(xy_label[1]) + h / 2) / 1080
            w = w / 1920
            h = h / 1080
            line_label_file = xy_label[-1] + " " + str(x1) + " " + str(y1) + " " + str(w) + " " + str(h) + "\n"
            label_file.write(line_label_file)
        label_file.close()






    text_file.close()
    text_file_2017.close()



def rename():
    path="C:/Users/YueBao/Desktop/bend/"
    filenames = os.listdir(path)
    for i, filename in enumerate(filenames):
        if not os.path.isfile(path+filename) and filename[-4:]!=".jpg":
            continue
        os.rename(path+filename, path+str(i)+".webp")
        xmlname = ""
        if filename.find(".jpg")!=-1:
            xmlname = filename[:filename.find(".jpg")]+".xml"
        if filename.find(".webp")!=-1:
            xmlname = filename[:filename.find(".webp")]+".xml"
        # os.rename(path+"labels/" + xmlname, path+"labels/" + str(i) + ".xml")


def change_small_txt2yolov5_need_txt():
    # img_paths = ["D:/paCong/rope_pet_labeld/", "D:/paCong/new_data_from_video1/"]
    img_paths = ["D:\paCong\jiankong\jiankong_video_data/3/images/"]
    label_path = "D:\paCong\jiankong\jiankong_video_data/3/labels/"
    new_img_path = "/home/mopanzhong/rope_detection/yolov5-master/data/rope_dog/jiankong_data/3/images/"
    # txt_path = "./xml2txt/"
    # img_name_txt = open("img_name_txt.txt", "w")
    for img_path in img_paths:
        img_name_txt = open(img_path+"img_name_txt.txt", "w")
        txt_path = img_path+"xml2txt/"
        if not os.path.exists(txt_path):
            os.mkdir(txt_path)
        img_names = os.listdir(img_path)
        for index_img_name, img_name in enumerate(img_names):
            index_point_img_name = img_name.find(".")
            # print(img_name[index_point_img_name:]!=".jpg")
            if (not os.path.isfile(img_path + img_name)) or img_name[index_point_img_name:]!=".jpg":
                continue

            img_name_txt.write(new_img_path + img_name+"\n")
            # print(img_path+img_name+"！！！！！！！！！！！")
            img_size = Image.open(img_path+img_name).size
            img_wid = img_size[0]
            img_hig = img_size[1]
            print(index_img_name, img_path+img_name, label_path+img_name[:index_point_img_name]+".xml")
            if os.path.exists(label_path+img_name[:index_point_img_name]+".xml"):
                tree = ET.parse(label_path+img_name[:index_point_img_name]+".xml")
                txt_file = open(txt_path+img_name[:index_point_img_name]+".txt", "w")
                for obj in tree.iter('object'):
                    label = 0
                    if obj.findtext('name') == "rope":
                        label = 1
                    elif obj.findtext('name') == "person":
                        label = 2
                    xmin = int(obj.findtext('bndbox/xmin'))
                    ymin = int(obj.findtext('bndbox/ymin'))
                    xmax = int(obj.findtext('bndbox/xmax'))
                    ymax = int(obj.findtext('bndbox/ymax'))

                    w = float(xmax - xmin)
                    h = float(ymax - ymin)
                    x1 = (float(xmin) + w / 2) / img_wid
                    y1 = (float(ymin) + h / 2) / img_hig
                    w = w / img_wid
                    h = h / img_hig

                    txt_file.write(str(label) + " " + str(x1) + " " + str(y1) + " " + str(w) + " " + str(h) + "\n")
                txt_file.close()
    img_name_txt.close()


def change_red_while():
    path = "D:\paCong/rope_pet_labeld\label_mask\mask/"
    new_path = path+"white/"
    # img_path = "D:\paCong\new_data_from_video1\mask_label\mask"
    num = 0
    filenames = os.listdir(path)

    for i, filename in enumerate(filenames):
        if not os.path.isfile(path+filename):
            continue
        # if os.path.exists(img_path+filename[:-3]+"jpg"):
        #     num+=1
        mask_sma = cv2.imread(path + filename)
        red_change_while = mask_sma>0
        red_change_while_res = np.logical_or(red_change_while[:,:,0], red_change_while[:,:,1])
        red_change_while_res = np.logical_or(red_change_while_res, red_change_while[:,:,2])
        red_change_while[:, :, 0] = red_change_while_res
        red_change_while[:, :, 1] = red_change_while_res
        red_change_while[:, :, 2] = red_change_while_res
        mask_sma[red_change_while] = 255
        cv2.imwrite(new_path+filename.replace(".png", ".jpg"), mask_sma)

    print(num)


def txt2yaml(path, new_path):
    txt = open(path, "r")
    yaml = open(new_path, "w")

    lines = "label_names:\n"
    line = txt.readline()
    while line:
        lines+="- "+line
        line = txt.readline()

    yaml.write(lines)
    txt.close()
    yaml.close()


def change_json2img():
    import os, shutil
    cmd = "labelme_json_to_dataset "
    img_path = "D:\paCong/rope_pet_labeld/"
    path = img_path+"label_mask/"
    newpath = path + "mask/"
    new_path_yaml = path+"yaml/"
    for root, dirs, files in os.walk(path):
        print(len(files))
        for i, name in enumerate(files):
            if not os.path.isfile(path+name):
                continue
            if not os.path.exists(img_path+name[:-4]+"jpg"):
                continue
            print(i, name)

            os.system(cmd + path + name)
            shutil.move(path + name.replace(".", "_") + "/label.png", newpath + name.replace("json", "png"))
            # txt2yaml(path + name.replace(".", "_") + "/label_names.txt",
            #          new_path_yaml + name.replace("json", "yaml"))


            # if " " in name:
            #     new_name = name.replace(" ", "_")
            #     try:
            #         os.rename(path + name, path + new_name)
            #     except:
            #         os.remove(path + new_name)
            #         os.rename(path + name, path + new_name)
            #     os.system(cmd + path + new_name)
            #     shutil.move(path + new_name.replace(".", "_") + "/label.png", newpath + name.replace("json", "png"))
            #     txt2yaml(path + new_name.replace(".", "_") + "/label_names.txt", new_path_yaml + name.replace("json", "yaml"))
            # else:
            #     os.system(cmd + path + name)
            #     shutil.move(path + name.replace(".", "_") + "/label.png", newpath + name.replace("json", "png"))
            #     txt2yaml(path + name.replace(".", "_") + "/label_names.txt",
            #              new_path_yaml + name.replace("json", "yaml"))


def get():
    img_path="D:\paCong\label/"

    filenames = os.listdir(img_path)
    for i, filename in enumerate(filenames):
        if not os.path.isfile(img_path+filename):
            continue
        if " " in filename:
            os.rename(img_path+filename, img_path+filename.replace(" ", "_"))




if __name__ == '__main__':
    change_big_txt2yolov5_need_txt()







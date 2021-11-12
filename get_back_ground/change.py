import os
import xml.etree.ElementTree as ET
import PIL.Image as Image

def change_big_txt2yolov5_need_txt():
    text_file = open("2007_val.txt", "r")
    text_file_2017 = open("val2017.txt", "w")
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


def change_small_txt2yolov5_need_txt():
    img_paths = ["D:/paCong/rope_pet_labeld/", "D:/paCong/new_data_from_video1/"]
    label_path = "D:/paCong/label/"
    new_img_path = "d:/pacong/"
    txt_path = "./xml2txt/"
    img_name_txt = open("img_name_txt.txt", "w")
    for img_path in img_paths:
        img_names = os.listdir(img_path)
        for index_img_name, img_name in enumerate(img_names):
            img_name_txt.write(new_img_path + img_name+"\n")
            img_size = Image.open(img_path+img_name).size
            img_wid = img_size[0]
            img_hig = img_size[1]
            print(index_img_name, img_path+img_name, label_path+img_name[:-4]+".xml")
            if os.path.exists(label_path+img_name[:-4]+".xml"):
                tree = ET.parse(label_path+img_name[:-4]+".xml")
                txt_file = open(txt_path+img_name[:-4]+".txt", "w")
                for obj in tree.iter('object'):
                    label = 0
                    if obj.findtext('name') == "rope":
                        label = 1
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

if __name__ == '__main__':
    change_small_txt2yolov5_need_txt()







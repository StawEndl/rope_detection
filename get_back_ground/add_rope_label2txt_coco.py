import os
import xml.etree.ElementTree as ET
import PIL.Image as Image

import csv

def change_xml2yolov5_txt():
    img_path = "D:\paCong\helpmet\helmet1\VOC2028/JPEGImages/"
    xml_path = "D:\paCong\helpmet\helmet1\VOC2028/Annotations/"
    txt_path = "D:\paCong\helpmet\helmet1\VOC2028/labels/"

    txt_names = open(txt_path+"filenames.txt","w")
    xml_names = os.listdir(xml_path)
    labels = []

    for i, xml_name in enumerate(xml_names):
        tree = ET.parse(xml_path+xml_name)
        txt = open(txt_path+xml_name[:-3]+"txt","w")
        line = ""

        print(i, xml_name)

        img_name = img_path + xml_name[:-3]+"jpg"
        img_size = Image.open(img_name).size
        txt_names.write(img_name+"\n")
        img_wid = img_size[0]
        img_hig = img_size[1]

        for obj in tree.iter('object'):
            label = obj.findtext('name')
            if label not in labels:
                labels.append(label)
            #print(label, label == "helmet", label == "hat")
            if label == "helmet" or label == "hat":
                label = "1"
            elif label=="person":
                continue
            else:
                label = "0"


            xmin = float(obj.findtext('bndbox/xmin'))
            ymin = float(obj.findtext('bndbox/ymin'))
            xmax = float(obj.findtext('bndbox/xmax'))
            ymax = float(obj.findtext('bndbox/ymax'))

            w = float(xmax - xmin)
            h = float(ymax - ymin)
            x1 = (float(xmin) + w / 2) / img_wid
            y1 = (float(ymin) + h / 2) / img_hig
            w = w / img_wid
            h = h / img_hig

            # print("1 "+str(x1)+" "+str(y1)+" "+str(w)+" "+str(h)+"\n")
            line += label+" "+str(x1)+" "+str(y1)+" "+str(w)+" "+str(h)+"\n"
        #print(line)
        txt.write(line)
        txt.close()
    print(labels)


def read_csv():
    img_path = "D:\paCong\helpmet\helmet_dataset\JPEGImages/"
    new_path = "d:/"
    txt_path = "./labels/"
    csvFile = open("train_labels.csv", "r")
    reader = csv.reader(csvFile)
    name_file = open("name.txt", "w")
    index = 0


    for data in reader:
        print(index, data)
        index+=1
        name, da = data[0], data[1]
        if not os.path.exists(img_path+name):
            continue
        name_file.write(new_path+name+"\n")

        img_name = img_path + name
        img_size = Image.open(img_name).size
        img_wid = img_size[0]
        img_hig = img_size[1]

        if da[-1]==" ":
            da = da[:-1]
        da = da.split(" ")
        i = 0
        txt_file = open(txt_path+name[:-3]+"txt", "w")
        lines = ""
        while i<len(da):
            line = ""
            label = "0 "

            #print(da[i])
            xmin = float(da[i])
            i += 1
            ymin = float(da[i])
            i += 1
            xmax = float(da[i])
            i += 1
            ymax = float(da[i])
            i += 1
            w = float(xmax - xmin)
            h = float(ymax - ymin)
            x1 = (float(xmin) + w / 2) / img_wid
            y1 = (float(ymin) + h / 2) / img_hig
            w = w / img_wid
            h = h / img_hig

            if da[i]!="none":
                label = "1 "

            lines+=label+str(x1)+" "+str(y1)+" "+str(w)+" "+str(h)+"\n"
            i+=1
        txt_file.write(lines)
        txt_file.close()

    name_file.close()



img_path = "D:\paCong\helpmet\helmet1\VOC2028\JPEGImages/"
xml_path = "D:\paCong\helpmet\helmet1\VOC2028\Annotations/"
txt_path = "D:\paCong\helpmet\helmet1\VOC2028/labels/"

txt_names = open(txt_path+"filenames.txt","w")
xml_names = os.listdir(xml_path)

for i, xml_name in enumerate(xml_names):
    tree = ET.parse(xml_path+xml_name)
    txt = open(txt_path+xml_name[:-3]+"txt","w")
    line = ""

    print(i, xml_name)

    img_name = img_path + xml_name[:-3]+"jpg"
    img_size = Image.open(img_name).size
    txt_names.write(img_name+"\n")
    img_wid = img_size[0]
    img_hig = img_size[1]

    for obj in tree.iter('object'):
        label = obj.findtext('name')
        if label=="helmet":
            label = "1"
        else:
            label = "0"
        xmin = float(obj.findtext('bndbox/xmin'))
        ymin = float(obj.findtext('bndbox/ymin'))
        xmax = float(obj.findtext('bndbox/xmax'))
        ymax = float(obj.findtext('bndbox/ymax'))

        w = float(xmax - xmin)
        h = float(ymax - ymin)
        x1 = (float(xmin) + w / 2) / img_wid
        y1 = (float(ymin) + h / 2) / img_hig
        w = w / img_wid
        h = h / img_hig

        # print("1 "+str(x1)+" "+str(y1)+" "+str(w)+" "+str(h)+"\n")
        line += label+" "+str(x1)+" "+str(y1)+" "+str(w)+" "+str(h)+"\n"
    #print(line)
    txt.write(line)


    txt.close()


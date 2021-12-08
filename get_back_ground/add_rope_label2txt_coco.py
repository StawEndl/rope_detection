import os
import xml.etree.ElementTree as ET
import PIL.Image as Image

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
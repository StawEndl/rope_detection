import os,shutil



path = "D:/paCong/coco/labels/val2017/"
img_path = "D:/paCong/val2017/"
new_path = "D:/paCong/person_dog_for_val/"
filenames = os.listdir(path)
num = 0

person_txt = open(new_path+"img/person_dog.txt", "w")

for index, filename in enumerate(filenames):
    # if num==5500:
    #     break
    new_txt = None
    txt = open(path + filename, "r")
    da = txt.readline()
    while da:
        if int(da.split(" ")[0])==0 or int(da.split(" ")[0])==16:
            if new_txt is None:
                new_txt = open(new_path+"labels/"+filename, "w")
            if int(da.split(" ")[0]) == 0:
                new_txt.write("2"+da[1:])
            elif int(da.split(" ")[0])==16:
                new_txt.write("0" + da[2:])
        da = txt.readline()
    print(len(filenames), index, new_txt is not None)
    if new_txt is not None:
        new_txt.close()
        shutil.copy(img_path+filename[:-4]+".jpg", new_path+"img/"+filename[:-4]+".jpg")
        person_txt.write(new_path+"img/"+filename[:-4]+".jpg\n")
    txt.close()
    num += 1
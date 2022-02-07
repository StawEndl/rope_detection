import os,shutil



path = "D:/paCong/person_dog/labels/"
new_path = "D:/paCong/person_dog/labels1/"
filenames = os.listdir(path)

for index, filename in enumerate(filenames):
    # if num==5500:
    #     break
    txt = open(path + filename, "r")
    new_txt = open(new_path + filename, "w")
    da = txt.readline()
    while da:
        line = ""
        if int(da.split(" ")[0])==6:
            line = "0"+da[2:]
        else:
            line = da
        new_txt.write(line)
        da = txt.readline()
    print(len(filenames), index)
    new_txt.close()
    txt.close()
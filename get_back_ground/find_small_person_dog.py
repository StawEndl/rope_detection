import os,shutil


def find():
    path = "D:/paCong/person_dog/labels/"
    img_path = "D:/paCong/person_dog/img/"
    new_path = "D:/paCong/person_dog/big_person/"
    filenames = os.listdir(path)
    max_area = 0.009

    person_txt = open(new_path+"person.txt", "w")
    # dog_txt = open(new_path + "dog.txt", "w")
    num_person = 0
    num_dog = 0


    for index, filename in enumerate(filenames):
        have_dog = False
        # if num==5500:
        #     break
        line = 1
        txt = open(path + filename, "r")
        da = txt.readline()
        num_person1 = 0
        num_dog1 = 0
        while da:
            d = da.split(" ")
            label, w, h = int(d[0]), float(d[3]), float(d[4])
            if label == 0:
                have_dog = True
                if w*h>=max_area/2:
                    # line += da
                    num_dog1 += 1
                    have_dog = True
                else:
                    line = None
                    break
                # num_dog1 += 1
                # have_dog = True
            elif label == 2:
                if w * h >= max_area:
                    # line += da
                    num_person1 += 1
                else:
                    line = None
                    break
            da = txt.readline()

        if line:
            num_person += num_person1
            num_dog += num_dog1
            # shutil.copy(path+filename, new_path+"labels/"+filename)
            if not have_dog:
                shutil.copy(img_path + filename[:-3] + "jpg", new_path + "images/" + filename[:-3] + "jpg")
                shutil.copy(path + filename, new_path + "labels/" + filename)
            # if have_dog:
            #     shutil.copy(img_path + filename[:-3] + "jpg", new_path + "images/dog/" + filename[:-3] + "jpg")
            # else:
            #     shutil.copy(img_path + filename[:-3]+"jpg", new_path + "images/" + filename[:-3]+"jpg")
        print(len(filenames), index, num_person, num_dog)
        txt.close()
        person_txt.write(new_path+filename[:-3]+"jpg")

    # num_txt.write("person:"+str(num_person)+"    dog:"+str(num_dog))
    # num_txt.close()
    person_txt.close()


def find2():
    path = "D:/paCong/person_dog/labels/"
    img_path = "D:/paCong/person_dog/small_person_dog/images/dog/"
    filenames = os.listdir(img_path)

    max_area = 0.0756

    for i, filename in enumerate(filenames):
        if os.path.isdir(img_path+filename):
            continue
        print(len(filenames), i, filename)
        txt = open(path+filename[:-3]+"txt", "r")
        da = txt.readline()
        is_small = False

        while da:
            d = da.split(" ")
            label, w, h = int(d[0]), float(d[3]), float(d[4])
            if w*h<max_area:
                is_small = True
                break
            da = txt.readline()
        if is_small:
            shutil.copy(img_path+filename, img_path+"ss/"+filename)
        else:
            shutil.copy(img_path + filename, img_path + "sb/" + filename)
        txt.close()


def test():
    path = "D:/paCong/person_dog/big_person/images/"
    filenames = os.listdir(path)
    txt = open(path+"files.txt", "w")

    for i, filename in enumerate(filenames):
        txt.write(path+filename+"\n")

    txt.close()

if __name__ == '__main__':
    test()
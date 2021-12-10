import cv2, os
from matplotlib import pyplot as plt
import shutil
# cap = cv2.VideoCapture("D:/paCong/bandicam 2021-10-18 17-07-19-842.mp4")

# ret, frame = cap.read()
#
# img = cv2.imread('D:/paCong/mask/new_photo41#1#.png')#0-400 0-300
# plt.imshow(frame)
# plt.show()
# if cv2.waitKey() == ord("q"):
#     cv2.destroyAllWindows()
def video_split():

    videos_path = "D:\paCong\person_fall_down/video/"
    save_path = videos_path+"img/"

    # video_names = os.listdir(videos_path)
    video_names = ["1.mp4","2.mp4","3.mp4","4.mp4","5.mp4","6.mp4","7.mp4","8.mp4","9.mp4",]
    # video_names = ["4.mp4"]
    for index_video_name, video_name in enumerate(video_names):
        #print(index_video_name)
        if index_video_name<1:
            continue
        cap = cv2.VideoCapture(videos_path+video_name)
        # print(video_name)
        i = 0
        while True:
            print(video_name, i)
            ret, frame = cap.read()
            if not ret:
                break
            i += 1
            if i%2!=0:
                continue

            if not os.path.exists(save_path+str(index_video_name)):
                os.mkdir(save_path+str(index_video_name))
            cv2.imwrite(save_path+str(index_video_name)+"/jiankong_video_data"+str(index_video_name)+"_"+str(i)+".jpg", frame)
            if cv2.waitKey() == ord("q"):
                cv2.destroyAllWindows()
        cap.release()

def copy():
    xml_patth = "D:/paCong/jiankong/jiankong_video_data/3/labels/"
    img_path = "D:/paCong\jiankong\jiankong_video_data/3/"
    new_path = "D:/paCong\jiankong\jiankong_video_data/3/images/"

    filenames = os.listdir(xml_patth)

    if not os.path.exists(new_path):
        os.mkdir(new_path)

    for i, filename in enumerate(filenames):
        if not os.path.isfile(xml_patth+filename) or filename[-4:]!=".xml":
            continue
        print(img_path+filename.replace(".xml", ".jpg"), new_path+filename.replace(".xml", ".jpg"))
        shutil.copy(img_path+filename.replace(".xml", ".jpg"), new_path+filename.replace(".xml", ".jpg"))


def rename():
    img_path = "D:\paCong\person_fall_down/"
    filenames = os.listdir(img_path)

    for i, filename in enumerate(filenames):
        if not os.path.isfile(img_path+filename):
            continue
        os.rename(img_path+filename, img_path+str(i)+".jpg")

copy()



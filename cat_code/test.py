import cv2, os
from matplotlib import pyplot as plt
# cap = cv2.VideoCapture("D:/paCong/bandicam 2021-10-18 17-07-19-842.mp4")

# ret, frame = cap.read()
#
# img = cv2.imread('D:/paCong/mask/new_photo41#1#.png')#0-400 0-300
# plt.imshow(frame)
# plt.show()
# if cv2.waitKey() == ord("q"):
#     cv2.destroyAllWindows()

videos_path = "D:/paCong/add_data_video/"
save_path = "D:/paCong/new_data_from_video/"

video_names = os.listdir(videos_path)
for index_video_name, video_name in enumerate(video_names):
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
        cv2.imwrite(save_path+str(index_video_name)+"/new_data_from_video"+str(index_video_name)+"_"+str(i)+".jpg", frame)
        if cv2.waitKey() == ord("q"):
            cv2.destroyAllWindows()
    cap.release()
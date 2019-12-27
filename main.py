from classify import Classify
import os
import facenet


modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl'
img_path = './test/MVIMG_20191114_125118_1.jpg'
img_dir = './test'

video_path = './test/video/test_video3.mp4'
cfy = Classify(modeldir, classifier_filename)
# text = [result_names, text_x, text_y, best_class_probabilities]   this is text_array result
total_img = 0
correct_count = 0
wrong_count = 0
confused_count=0
cannot_classify_count = 0
not_confused_count = 0
# not confused      0
# confused          1
# cannot classify   2

#AMBER
# classify images in folder test
for imgName in os.listdir(img_dir):
    text = cfy.classify_image(os.path.join(img_dir, imgName))
    # print("TEXT:")
    print(text[0])  #this is text_array
    # print(text[0][0])  #predicted name
    # print(text[0][3][0])  #predicture certainty, might not need this
    # print(text[0][5])  #confusion
    total_img = total_img +1
    if text[0][5] == 0:
        not_confused_count = not_confused_count + 1
    if text[0][5] == 1:
        confused_count = confused_count +1
    if text[0][5] == 2:
        cannot_classify_count = cannot_classify_count + 1
    if imgName.find(text[0][0])== -1:
        wrong_count = wrong_count +1
    else:
        correct_count= correct_count + 1
print("total_img: ",total_img)
print("correct_count: ",correct_count)
print("wrong_count: ",wrong_count)
print("confused_count: ",confused_count)
print("cannot_classify_count: ",cannot_classify_count)
print("not_confused_count: ",not_confused_count)



    

# dataset = facenet.get_dataset(img_dir)
# for img in dataset:
#     cfy.classify_image(img)
    

    

# classify images 
# cfy.classify_image(img_path)


# classify video
# cfy.classify_video(video_path)

# classify webcam
# cfy.classify_webcam()

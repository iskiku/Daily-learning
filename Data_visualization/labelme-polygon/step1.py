"""
@Time ： 2024/9/12 11:09
@Author ： jtch
"""
import os
import shutil

# 存放图片和json的路径
JPG_folder = r" "
# 保存原始图片路径
Paste_JPG_folder = r"D:\22"
# 保存二值掩码图片路径
Paste_label_folder = r"D:\33"
# 保存带有标签的图像路径
Paste_label_viz_folder = r"D:\44"

folder = [JPG_folder,Paste_JPG_folder, Paste_label_folder, Paste_label_viz_folder]
for folder_x in folder:
    if not os.path.exists(folder_x):
        os.makedirs(folder_x)

#  获取文件夹内的文件名
FileNameList = os.listdir(JPG_folder)
NewFileName = 1
for i in range(len(FileNameList)):
    #  判断当前文件是否为json文件
    if(os.path.splitext(FileNameList[i])[1] == ".jpg"):

        #  复制jpg文件
        JPG_file = JPG_folder + "\\" + FileNameList[i]
        new_JPG_file = Paste_JPG_folder + "\\" + str(NewFileName) + ".jpg"
        shutil.copyfile(JPG_file, new_JPG_file)

        #  复制label文件
        jpg_file_name = FileNameList[i].split(".", 1)[0]
        label_file = JPG_folder + "\\" + jpg_file_name + "_json\\label.png"
        new_label_file = Paste_label_folder + "\\" + str(NewFileName) + ".png"
        shutil.copyfile(label_file, new_label_file)

        #  复制label_viz文件
        jpg_file_name = FileNameList[i].split(".", 1)[0]
        label_viz_file = JPG_folder + "\\" + jpg_file_name + "_json\\label_viz.png"
        new_label_viz_file = Paste_label_viz_folder + "\\" + str(NewFileName) + ".png"
        shutil.copyfile(label_viz_file, new_label_viz_file)

        #  文件序列名+1
        NewFileName = NewFileName + 1


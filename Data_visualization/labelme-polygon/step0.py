"""
@Time ： 2024/9/12 11:08
@Author ： jtch
"""
import os

# 存放图片和json的路径
json_folder = r""
#  获取文件夹内的文件名
FileNameList = os.listdir(json_folder)
#  激活labelme环境
os.system("activate labelme")
for i in range(len(FileNameList)):
    #  判断当前文件是否为json文件
    if (os.path.splitext(FileNameList[i])[1] == ".json"):
        json_file = json_folder + "\\" + FileNameList[i]
        #  将该json文件转为png
        os.system("labelme_json_to_dataset " + json_file)

import os
###### 将mp4批量转为mp3且统一名
for i in range(8):
    os.system("cd Friends && mv 老友记第"+str(i+3)+"季 S"+str(i+3))
    os.system("cd S"+str(i+3))
for i in range(8):
    for j in range(26):
        os.system("cd Friends && cd S"+str(i+3)+" && mv 老友记第"+str(i+3)+"季第"+str(j+1)+"集.mp4 "+str(i+3)+"-"+str(j+1)+".mp4")
for j in range(9):
    for i in range(26):
        os.system("cd Friends && cd S" + str(j+2) + " && ffmpeg -i " + str(j+2) + "-" + str(i + 1) + ".mp4 -f mp3 -vn y" + str(
            j+2) + "_" + str(i + 1) + ".mp3")
###### 统一字幕文件名
path_parent = 'Friends/srtfile'
path_parent_r = 'Friends/srtfile/'
f_parent = os.listdir(path_parent)
for i, old_name in enumerate(f_parent):
    # name_parent = old_name.replace('Friends-Season','S')
    # print(name_parent)
    os.rename(path_parent_r+old_name, path_parent_r+'S'+str(i+1))

# print(name_parent)
for i in range(10):
    path = 'Friends/srtfile/S'+str(i+1)
    path_r = path+'/'
    f = os.listdir(path)
    for j, old_name in enumerate(f):
        print(str(j+1),old_name)
        os.rename(path_r+old_name, path_r+'z'+str(i+1)+'_'+str(j+1)+'.srt')

import os
import re
# ###### 将mkv批量转为mp3且统一名
# for i in range:
#     os.system("cd Friends && mv S0"+str(i+1)+" S"+str(i+1))
# for i in range(8):
#     for j in range(26):
#         os.system("cd Friends && cd S"+str(i+3)+" && mv 老友记第"+str(i+3)+"季第"+str(j+1)+"集.mp4 "+str(i+3)+"-"+str(j+1)+".mp4")
# for j in range(9):
#     for i in range(26):
#         os.system("cd Friends && cd S" + str(j+2) + " && ffmpeg -i " + str(j+2) + "-" + str(i + 1) + ".mp4 -f mp3 -vn y" + str(
#             j+2) + "_" + str(i + 1) + ".mp3")


###### 提取字幕文件
video_path = 'Friends'
save_path = video_path+'/srtfile'
save_path_ = save_path+'/'
for i in range(10):
    s_path = video_path+'/S'+str(i+1)
    s_path_ = s_path+'/'
    f_parent = os.listdir(s_path)
    srt_parent = save_path_+'S'+str(i+1)
    srt_parent_ = srt_parent+'/'
    isExists = os.path.exists(srt_parent)
    p = 0
    if not isExists:
        os.makedirs(srt_parent)
    for j, old_name in enumerate(f_parent):
        reg = re.compile(r'mkv$')
        if reg.search(old_name):
            # print(old_name)
            os.system('ffmpeg -i '+s_path_+old_name+' -map 0:6 '+srt_parent_+'S'+str(i+1)+'-'+str(p+1)+'.srt')
            p += 1
        # os.system('ffmpeg -i '+s_path_+old_name+' -f mp3 -vn '+s_path_+'S'+str(i+1)+'-'+str(j+1)+'.mp3')
        # os.rename(s_path_+old_name, s_path_+'S'+str(i+1)+'-'+str(j+1))




# ###### 统一字幕文件名
# path_parent = 'Friends/srtfile'
# path_parent_r = 'Friends/srtfile/'
# f_parent = os.listdir(path_parent)
# for i, old_name in enumerate(f_parent):
#     # name_parent = old_name.replace('Friends-Season','S')
#     # print(name_parent)
#     os.rename(path_parent_r+old_name, path_parent_r+'S'+str(i+1))
#
# # print(name_parent)
# for i in range(10):
#     path = 'Friends/srtfile/S'+str(i+1)
#     path_r = path+'/'
#     f = os.listdir(path)
#     for j, old_name in enumerate(f):
#         print(str(j+1),old_name)
#         os.rename(path_r+old_name, path_r+'z'+str(i+1)+'_'+str(j+1)+'.srt')

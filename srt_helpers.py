from srt_helpers import srtParsing
import codecs
import os
# srt = srtParsing
file = codecs.open("test1.srt",'r','utf-8')
# print(file)
start,duration,text = srtParsing().srtProcess(file)
# print(len(aa),len(bb),len(cc))
parent_path = 'audio_cut_test/'
path = parent_path + 'test11'
isExists = os.path.exists(path)
if not isExists:
    os.makedirs(path)
    print('创建'+path+'目录成功')
else:
    print(path+'目录已存在')
for i in range(len(start)):
    os.system('cd audio_cut_test && ffmpeg -i %s -ss %d -t %d -codec copy %s'%('test11.mp3',start[i],duration[i],'test11/test11_'+str(i)+'.mp3'))

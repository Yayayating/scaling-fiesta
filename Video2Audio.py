import os
j = 1
for i in range(25):
    os.system("cd Friends && cd S"+str(j)+" && ffmpeg -i "+str(j)+"-"+str(i+1)+".mp4 -f mp3 -vn y"+str(j)+"_"+str(i+1)+".mp3")
    

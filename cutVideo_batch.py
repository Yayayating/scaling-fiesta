import os
import json
# save_file_name = []
# with open("ghost_in_shell/",'r') as save_file:
#     for row in save_file:
#         save_file_name.append(row.replace('\n',''))
# print(save_file_name)
start = []
with open("ghost_in_shell/start_time_gis.txt",'r') as time_start:
    for row in time_start:
        start.append(row.replace('\n',''))
print(start)
end = []
with open("ghost_in_shell/end_time_gis.txt",'r') as time_end:
    for row in time_end:
        end.append(row.replace('\n',''))
# print(end)
for i in range(len(end)):
    os.system("cd ghost_in_shell && ffmpeg -ss %s -t %s -i Ghost.in.the.Shell.mkv -codec copy %s"%(start[i],end[i],'ghost_in_the_shell_'+str(i)+'.mkv'))

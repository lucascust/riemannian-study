import csv
import numpy as np

with open("./data/Dataset 1/A001SB1_1.csv") as csv_file:
    lines = csv_file.readlines()
    header = lines.pop(0)
    

    #Formata em "Timestamp[Valores]""
    for i in range(len(lines)):
        lines[i] = lines[i].split(",")
        lines[i][1] = lines[i][1:] 
        lines[i][1][-1] = lines[i][1][-1].rstrip("\n") 
        lines[i] = lines[i][0:2]
    
    lines = np.array(lines)
    for i in lines:
        i = i.astype(np.float)
    print("pausa")


    newfile = open("newdata.txt","w+")
    for i in range(len(lines)):
        newfile.write(''.join(str(e) for e in lines[i]) + '\n' )

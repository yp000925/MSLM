import numpy as np
from lxml import etree
import csv

mydoc = etree.parse("./VeRi_with_plate/train_label.xml")
imageName=[]
vehicleID=[]
cameraID=[]
root = mydoc.getroot()
items = root[0]
for item in items:
    imageName.append(item.attrib['imageName'])
    vehicleID.append(item.attrib['vehicleID'])
    cameraID.append(item.attrib['cameraID'])

with open('./VeRi_with_plate/veri_train.txt','w') as file:
    for i,v,c in zip(imageName,vehicleID,cameraID):
        file.writelines([i,' ',v,' ',c,'\n'])

with open('./VeRi_with_plate/veri_train.csv',mode = 'w') as csv_file:
    writer = csv.writer(csv_file,delimiter = ',')
    for fid,pid in zip(imageName,vehicleID):
        writer.writerow([pid,fid])

mydoc = etree.parse("./VeRi_with_plate/test_label.xml")
imageName=[]
vehicleID=[]
cameraID=[]
root = mydoc.getroot()
items = root[0]
for item in items:
    imageName.append(item.attrib['imageName'])
    vehicleID.append(item.attrib['vehicleID'])
    cameraID.append(item.attrib['cameraID'])

with open('./VeRi_with_plate/veri_test.txt','w') as file:
    for i,v,c in zip(imageName,vehicleID,cameraID):
        file.writelines([i,' ',v,' ',c,'\n'])

with open('./VeRi_with_plate/veri_test.csv',mode = 'w') as csv_file:
    writer = csv.writer(csv_file,delimiter = ',')
    for fid,pid in zip(imageName,vehicleID):
        writer.writerow([pid,fid])


with open('./VeRi_with_plate/veri_query.txt','w') as file:
    for i,v,c in zip(query_fid,pids,cameraids):
        file.writelines([i,' ',v,' ',c,'\n'])
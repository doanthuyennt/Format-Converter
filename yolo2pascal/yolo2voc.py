import sys

sys.path.insert(0,"/home/mmlab/git_workspace/FormatConverter/yolo2pascal/")
from pascal_voc_io import XML_EXT
from pascal_voc_io import PascalVocWriter
from yolo_io import YoloReader
import os.path
import sys
from tqdm import tqdm
from PyQt5.QtGui import QImage
from pathlib import Path


imgFolderPath = sys.argv[1]
known_img_format = ['.jpg','.jpeg','.png']
# Search all yolo annotation (txt files) in this folder
for file in tqdm(os.listdir(imgFolderPath)):
    if file.endswith(".txt") and file != "classes.txt":
        print("Convert", file)

        annotation_no_txt = os.path.splitext(file)[0]
        for format in known_img_format:
          imagePath = imgFolderPath + "/" + annotation_no_txt + format
          if Path(imagePath).exists():
            break    

        image = QImage()
        image.load(imagePath)
        imageShape = [image.height(), image.width(), 1 if image.isGrayscale() else 3]
        imgFolderName = os.path.basename(imgFolderPath)
        imgFileName = os.path.basename(imagePath)

        writer = PascalVocWriter(imgFolderName, imgFileName, imageShape, localImgPath=imagePath)


        # Read YOLO file
        txtPath = imgFolderPath + "/" + file
        tYoloParseReader = YoloReader(txtPath, image)
        shapes = tYoloParseReader.getShapes()
        num_of_box = len(shapes)

        for i in range(num_of_box):
            label = shapes[i][0]
            xmin = shapes[i][1][0][0]
            ymin = shapes[i][1][0][1]
            x_max = shapes[i][1][2][0]
            y_max = shapes[i][1][2][1]

            writer.addBndBox(xmin, ymin, x_max, y_max, label, 0)

        writer.save(targetFile= imgFolderPath + "/" + annotation_no_txt + ".xml")

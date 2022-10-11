from PIL import Image,ImageDraw,ImageFont
# import matplotlib.pyplot as plt
import os
from sys import platform
# import numpy as np
import pathlib
import argparse

# command
# python .\font2img.py --ttf_path ..\..\..\..\Manga\fonts\ --img_size 64 --chara_size 64

parser = argparse.ArgumentParser(description='Obtaining characters from .ttf')
parser.add_argument('--ttf_path', type=str, default='../ttf_folder',help='ttf directory')
parser.add_argument('--chara', type=str, default='../chara.txt',help='characters')
parser.add_argument('--save_path', type=str, default='../save_folder',help='images directory')
parser.add_argument('--img_size', type=int, help='The size of generated images')
parser.add_argument('--chara_size', type=int, help='The size of generated characters')
args = parser.parse_args()

file_object = open(args.chara, encoding='utf-8')   
try:
	characters = file_object.read()
finally:
    file_object.close()

print(characters)

def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # get real offset according to font
    font_width, font_height = font.getsize(ch)
    font_offset = font.getoffset(ch)
    font_offset = (
        (canvas_size - font_width - font_offset[0])/2 ,
        (canvas_size - font_height - font_offset[1])/2, 
    )
    draw.text((x_offset+font_offset[0], y_offset+font_offset[1]), ch, (0, 0, 0), font=font)
    return img

def draw_example(ch, src_font, canvas_size, x_offset, y_offset):
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    example_img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    example_img.paste(src_img, (0, 0))
    return example_img

data_dir = args.ttf_path
data_root = pathlib.Path(data_dir)
print(data_root)

ignore_list = [
    "A-OTF-ShinMGoMin-Emboss-2", 
    "A-OTF-ShinMGoMin-Futoline-2", 
    "A-OTF-ShinMGoMin-Line-2", 
    "A-OTF-ShinMGoMin-Shadow-2", 
    "HonobonoPop-Regular", 
    "GN-Natsuiro_Schedule", 
    "KFhimajiWAKU", 
    "Pigmo-00", 
    "Pigmo-01", 
    "GNKana-Kiniro_SansSerif_ST", 
    "KFhimajiSTITCH", 
    "KouzanBrushFontSousyo", 
    "KouzanGyousho", 
    "KouzanMouhituFont", 
    "RiBenQingLiuHengShanMaoBiZiTi-2", 
    "kirin-Regular", 
]

def collect_font_path(extension):
    paths = []
    for ext in extension:
        tmp_paths = list(data_root.glob(f"*.{ext}*"))
        for path in tmp_paths:
            path = str(path)
            name = os.path.basename(path).split(".")[0]
            if name in ignore_list:
                continue
            paths.append(path)
    return paths

# all_image_paths = list(data_root.glob('*.ttf*'))
# all_image_paths = [str(path) for path in all_image_paths]
if platform == "win32":
    all_image_paths = collect_font_path(['otf', 'ttc', 'ttf'])
else:
    # assume the other system is case-insensitive
    all_image_paths = collect_font_path(['otf', 'ttc', 'ttf', 'TTF'])
print(len(all_image_paths))
# for i in range (len(all_image_paths)):
#     print(all_image_paths[i])

for (label,item) in zip(range(len(all_image_paths)), all_image_paths):
    print(os.path.basename(item))
    src_font = ImageFont.truetype(item, size = args.chara_size)
    for (chara,cnt) in zip(characters, range(len(characters))):
        img = draw_example(chara, src_font, args.img_size, (args.img_size-args.chara_size)/2, (args.img_size-args.chara_size)/2)
        path_full = os.path.join(args.save_path, 'id_%d'%label)
        if not os.path.exists(path_full):
            os.makedirs(path_full)
        img.save(os.path.join(path_full, "%04d.png" % (cnt)))
        
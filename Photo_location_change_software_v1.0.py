"""
著作財產權聲明

本聲明旨在明確聲明，本程序的所有內容均為作者獨家創作，並享有著作財產權。作者特此聲明，本程序內容不得被無法授權的第三方複製、修改、發布、傳播或以其他方式使用。

然而，作者在此同意，本程序內容可以根據知識共享（Creative Commons）授權條款進行共享。根據該條款，任何人都可以自由地複製、修改、發布和傳播本程序內容，但必須遵守以下條件：

無商業用途：本程序內容不得用於商業目的。
獨立作品：本程序內容不得與其他作品混雜或結合。
相同條款：任何基於本程序內容創作的衍生作品必須以相同的方式進行共享。
請注意，本聲明並不影響作者對本程序內容的著作財產權的保留。作者保留對本程序內容的修改、更新和撤回共享的權利。

本聲明自本程序首次發布之日起生效。作者保留在未經作者明確授權的情況下，修改本聲明的權利。

------------------------------------------------------

Copyright Statement

This statement clearly states that all contents of this program are exclusively created by the author and enjoy copyright. The author hereby declares that the contents of this program shall not be copied, modified, published, transmitted, or used in any other way by unauthorized third parties.

However, the author agrees to share the contents of this program under the Creative Commons license terms. According to the terms, anyone can freely copy, modify, publish, and transmit the contents of this program, but must comply with the following conditions:

Non-commercial use: The contents of this program shall not be used for commercial purposes.
Independent work: The contents of this program shall not be mixed or combined with other works.
Same terms: Any derivative works based on the contents of this program must be shared in the same way.
Please note that this statement does not affect the author's right to the copyright of the contents of this program. The author reserves the right to modify, update, and withdraw sharing of the contents of this program without the author's explicit permission.

This statement takes effect as of the first release of this program. The author reserves the right to modify this statement without the author's explicit permission.

聲明人 Statement by: wangsan71
聲明日期 Statement Date: 2024-11-15
源文件地址 Source file address: https://github.com/wangsan71/Change_Photo_XY
"""

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from tkinter import filedialog
import os
from xy_dataset import XYDataset
import cvui
import random, time

WINDOW_NAME = "Photo location change software v1.0 by WangSan"

path_of_original = ''
path_of_new = ''

pathfull_of_org = ''
pathfull_of_new = ''

img_file_list = []

img1, img2 = '', ''

hit_x, hit_y = 0, 0
"""
def torch_model_loader(model_name):
    import torch
    import torchvision
    from torch2trt import torch2trt
    from torch2trt import TRTModule
    from utils import preprocess

    CATEGORIES = ['apex']
    device = torch.device('cuda')
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 2 * len(CATEGORIES))
    model = model.cuda().eval().half()
    model.load_state_dict(torch.load(model_name))
    data = torch.zeros((1, 3, 224, 224)).cuda().half()
    model_trt = torch2trt(model, [data], fp16_mode=True)
    torch.save(model_trt.state_dict(), 'road_following_model_trt.pth')
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load('road_following_model_trt.pth'))  

def torch_model_predict():
    image = camera.read()
    image = preprocess(image).half()
    output = model_trt(image).detach().cpu().numpy().flatten()
    global x_out, y_out
    x_out = float(output[0])
    y_out = float(output[1])
"""
def Load_folder_photos():
    global path_of_original, pathfull_of_org
    global path_of_new, pathfull_of_new
    path_of_original = tk.filedialog.askdirectory(title="請選擇原始資料夾")
    path_of_new = tk.filedialog.askdirectory(title="請選擇新保存資料夾")
    print(os.path.abspath(path_of_original))
    print(os.path.abspath(path_of_new))

def get_photo(img_file):
    """
    TODO Base Mode:
    這裏的代碼是用來讀取圖片的
    還有更改頭文件的坐標
    如XXX_XXX_.....jpg
    把文件頭的第一和第二數字組更改就是XY軸
    """
    # file_jpg = tk.filedialog.askopenfilename(filetypes = [('JPG File', '*.jpg')])
    # print(file_jpg)

    # file = os.listdir(img_file)
    file = os.path.basename(img_file)
    file = os.path.splitext(file)[0]
    file = file.split('_')
    # print(file)
    x, y = int(file[0]), int(file[1])
    # file_jpg = os.path.basename(file_jpg)
    # # print(file_jpg)
    # file = os.path.splitext(file_jpg)[0]
    # file = file.split('_')
    # x, y  = file[0], file[1]
    # print(x, y)

    # dataset = XYDataset(file_jpg)
    # x, y = dataset.get_xy()
    # x, y = int(x), int(y)
    # print(x, y)
    return x, y

def image_of_saver(path_of_original):
    global img1, img2, img_file_list
    for file in os.listdir(path_of_original):
        full_path = os.path.join(path_of_original, file)
        # x, y = get_photo(full_path)
        img = cv2.imread(full_path)
        if img is not None:
            img_file_list.append(full_path)

            # print(len(img_file_list))
            # print(img_file_list)
def main():
    photo = 0
    frame = np.zeros((255, 550, 3), np.uint8)
    cvui.init(WINDOW_NAME)

    Load_folder_photos()
    image_of_saver(path_of_original)

    def start_image():
        global img1, img, img2
        img = cv2.imread(img_file_list[photo])
        img1 = img.copy()  
        img1 = cv2.circle(img1, (get_photo(img_file_list[photo])[0], get_photo(img_file_list[photo])[1]), 5, (0, 0, 255), 1) 
        img2 = img.copy()
        img2 = cv2.circle(img2, (get_photo(img_file_list[photo])[0], get_photo(img_file_list[photo])[1]), 5, (0, 0, 255), 1)
        
    start_image()
    mouse_down = False
    def mouse_callback(event, x, y, flags, userdata):
        nonlocal mouse_down
        global hit_x, hit_y
        if event == cv2.EVENT_LBUTTONDOWN and not mouse_down:
            mouse_down = True
            hit_x, hit_y = x, y
            global img2
            start_image()
            cv2.circle(img2, (x, y), 5, (255, 0, 0), 1)
        elif event == cv2.EVENT_LBUTTONUP and mouse_down:
            mouse_down = False

    cv2.namedWindow(WINDOW_NAME+"original", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME+"original", mouse_callback)

    while True:
        frame[:] = (49, 52, 49)
        
        cvui.text(frame, 300, 10, "Press ESC to exit.")
        cvui.text(frame, 300, 25, "Press 'Save' button than")
        cvui.text(frame, 300, 40, "you need to press'next' button")
        cvui.text(frame, 300, 55, "again to the next photo.")
        cvui.text(frame, 300, 80, f"No. {photo+1} / {len(img_file_list)} ; {hit_x}, {hit_y}")
        
        if cvui.button(frame, 300, 120, "Click to save"):
            name=''.join(random.choices(string.ascii_letters + string.digits, k=12))
            cv2.imwrite(path_of_new + '/' + str(hit_x) + '_' + str(hit_y) + '_' + str(name) + '.jpg', img)
        
        if cvui.button(frame, 300, 150, "Click to back photo"):
            photo -= 1 if photo > 0 else 0
            start_image()
        if cvui.button(frame, 300, 180, "Click to next photo"):
            photo += 1 if photo < len(img_file_list) - 1 else 0
            start_image()

        if cvui.button(frame, 300, 210, "Click to Exit"):
            break

        cvui.image(frame, 0, 0, img2)
        cv2.imshow(WINDOW_NAME+"original", img1)

        cvui.update()
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
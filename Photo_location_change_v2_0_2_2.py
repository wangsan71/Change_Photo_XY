"""
源文件地址 Source file address: https://github.com/wangsan71/Photo_location_change

V2.0.2 Update: 2024-11-19 19:27 微升級版 Little update
    1. 修正在保存圖片時沒有預測的問題
    2. 正式確定了在控制台視窗的圖片預覽會刪掉，在此版本中已經修正

V2.0.1 Update: 2024-11-16 22:20 微升級版 Little update
    1. 在分頁視窗上做到了同步避免需要多次在兩個視窗上觀看不同的點位, 正在思考需不需要把兩個視窗改成一個視窗為圖片預覽和坐標點操作,
        另一個作為操控台點擊按鈕, 這樣就可以避免需要切換視窗的麻煩以及佔用空間

V2.0 Update: 2024-11-16 19:37 正式完成 Turely final version
    1. 為V1.1 & V1.0的改進, 添加了一些新功能以及模型預測功能正式完善
        1.1 修正了一些圖形預測功能BUG
        1.2 添加了"自動預測"功能, 可以選擇關閉或開啟
        1.3 添加保留模型預測坐標點功能, 避免在選擇新的坐標時被刷走而再次預測
        1.4 添加了再次預測功能, 可以在"自動預測"功能關閉時使用預測坐標點
        1.5 新增了"清除載入模型"按鈕, 以便更換模型
    2. 修正了一些BUG, 使其能夠正常運行

V1.1 Update: 2024-11-16 Beta version: 
    1. 新增了"模型預測"基礎功能，需要手動開啟才能預測
    2. 新增輔助線功能, 為其方便操作

    不足之處:
        1. 模型預測功能尚未完善
        2. 在選擇新的坐標時模型預測坐標點被刷走

V1.0 Update: 2024-11-16 01:31 正式發佈, 有許多BUG, 未完善!

"""

import platform
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog
import os
# from xy_dataset import XYDataset
import cvui
import random
import string

WINDOW_NAME = "Photo location change software v2.0.1 by WangSan"

path_of_original = ''
path_of_new = ''

pathfull_of_org = ''
pathfull_of_new = ''

img_file_list = []
img1, img2 = '', ''

hit_x, hit_y = -1, -1
x_out, y_out = -1, -1

model = None
device = None

nothing_useful = QApplication([])

def torch_model_loader(model_name):
    import torch
    import torchvision.models as models
    global model, device, model_bool_auto
    
    print(model_name)
    
    # 设置设备
    os_type = platform.system()
    if os_type == "Darwin":
        device = torch.device('cpu')
    elif os_type == "Windows":
        # 在Windows上，检查是否有可用的CUDA设备
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device('cpu')
    
    # 加载ResNet18模型，使用weights参数替代pretrained
    print("Reading model")
    model = models.resnet18(weights=None)
    
    # 修改最后一层全连接层以适应分类任务
    CATEGORIES = ['apex']
    model.fc = torch.nn.Linear(512, 2 * len(CATEGORIES))
    
    # 将模型移到CUDA并设置为评估模式
    print("Setting model to CUDA")
    model = model.to(device).eval()
    
    # 加载模型权重
    print("Loading model weights")
    try:
        model.load_state_dict(torch.load(model_name, map_location=device))
        model_bool_auto = True
    except Exception as e:
        print(f"Error loading model weights: {e}")
        model_bool_auto = False
        return None
    
    return model

def torch_model_predict(img_path):
    import torch
    from PIL import Image
    import torchvision.transforms as transforms
    global device, model, img2, hit_x, hit_y, x_out, y_out
    
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 檢查 img 的類型
    if isinstance(img_path, str):
        # 如果 img 是文件路徑，則打開圖像
        image = Image.open(img_path)
    elif isinstance(img_path, np.ndarray):
        # 如果 img 是 numpy 數組，則直接使用
        image = Image.fromarray(img_path)
    else:
        raise ValueError("img must be a file path or a numpy array")
    
    # 對圖像進行預處理
    print("Preprocessing image")
    image = preprocess(image).half().to(device)
    # 增加批次維度
    image = image.unsqueeze(0)
    
    if model.type == torch.float16:
        # 如果模型是半精度，确保输入也是半精度
        image = image.to(device).half()
    else:
        # 如果模型是全精度，确保输入也是全精度
        image = image.to(device).float()
    
    # 使用模型進行預測
    # print(type(model))

    with torch.no_grad():
        if model != None:
            output = model(image).detach().cpu().numpy().flatten()
        else:
            return
    
    # 提取預測結果
    x_out = int(112 + float(output[0]) * 112)
    y_out = int(112 + float(output[1]) * 112)
    
    # 假設 img2 是已經加載的圖像
    try:
        img2 = img2.copy()
    except:
        img2 = cv2.imread(img_path)
    finally:
        pass

    # 大窗户使用IMG2
    cv2.circle(img2, (get_photo(img_path)[0], get_photo(img_path)[1]), 5, (0, 0, 255), 1)
    if hit_x != get_photo(img_path)[0] and hit_y != get_photo(img_path)[1]:
        cv2.circle(img2, (hit_x, hit_y), 5, (255, 0, 0), 1)
    cv2.circle(img2, (int(x_out), int(y_out)), 5, (255, 0, 255), 2)
    # 小窗户使用IMG1
    cv2.circle(img1, (get_photo(img_path)[0], get_photo(img_path)[1]), 5, (0, 0, 255), 1)
    if hit_x != get_photo(img_path)[0] and hit_y != get_photo(img_path)[1]:
        cv2.circle(img1, (hit_x, hit_y), 5, (255, 0, 0), 1)
    cv2.circle(img1, (int(x_out), int(y_out)), 5, (255, 0, 255), 2)
    print(x_out, y_out)

def Load_folder_photos():
    global path_of_original, pathfull_of_org
    global path_of_new, pathfull_of_new
    path_of_original = QFileDialog.getExistingDirectory(None, "請選擇原本的資料夾/Choose the new folder for save images")
    path_of_new = QFileDialog.getExistingDirectory(None, "請選擇新保存資料夾/Choose the new folder for save images")
    print(os.path.abspath(path_of_original))
    print(os.path.abspath(path_of_new))

def get_photo(img_file):
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
    from tkinter import filedialog
    global hit_x, hit_y, model
    
    photo = 0
    frame = np.zeros((280, 300, 3), np.dtype('uint8'))
    cvui.init(WINDOW_NAME)

    Load_folder_photos()
    image_of_saver(path_of_original)

    def start_image():
        global img1, img, img2, hit_x, hit_y, x_out, y_out
        # 基本載入圖片
        img = cv2.imread(img_file_list[photo])
        img1 = img.copy()  
        img1 = cv2.circle(img1, (get_photo(img_file_list[photo])[0], get_photo(img_file_list[photo])[1]), 5, (0, 0, 255), 1) 
        
        # 画辅助线
        # 中心线
        cv2.line(img1,(0,112),(224,112),(177,221,225),1)
        cv2.line(img1,(112, 0),(112, 224),(177,221,225),1)

        # 归112 范围线
        cv2.line(img1,(0,117),(224,117),(93,110,225),1)
        cv2.line(img1,(0,107),(224,107),(93,110,225),1)

        #四分线
        cv2.line(img1,(0,56),(224,56),(250,181,161),1)
        cv2.line(img1,(0,163),(224,163),(250,181,161),1)
        
        img2 = img.copy()
        img2 = cv2.circle(img2, (get_photo(img_file_list[photo])[0], get_photo(img_file_list[photo])[1]), 5, (0, 0, 255), 1)
        # hit_x, hit_y = get_photo(img_file_list[photo])
        # hit_x, hit_y = -1, -1
        x_out, y_out = -1, -1
        
    start_image()
    mouse_down = False
    def mouse_callback(event, x, y, flags, userdata):
        global hit_x, hit_y, x_out, y_out, x_out_2, y_out_2
        if x_out != -1 and y_out != -1:
                x_out_2, y_out_2 = x_out, y_out
        elif x_out == -1 and y_out == -1:
            pass
        nonlocal mouse_down
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_down = True
            hit_x, hit_y = x, y
            global img2
            start_image()
            cv2.circle(img2, (x, y), 5, (255, 0, 0), 1)
            cv2.circle(img1, (x, y), 5, (255, 0, 0), 1)
            if x_out_2 != -1 and y_out_2 != -1:
                cv2.circle(img2, (int(x_out_2), int(y_out_2)), 5, (255, 0, 255), 2)
                cv2.circle(img1, (int(x_out_2), int(y_out_2)), 5, (255, 0, 255), 2)
        # elif event == cv2.EVENT_LBUTTONUP and mouse_down:
        #     mouse_down = False

    cv2.namedWindow(WINDOW_NAME+"original", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME+"original", mouse_callback)
    
    def hit_point_reset():
        global hit_x, hit_y, x_out_2, y_out_2
        hit_x, hit_y = get_photo(img_file_list[photo])
        x_out_2, y_out_2 = -1, -1
        print("hit point reset")
        
    hit_point_reset()
    button_name = "Load Model"
    model_bool_auto = False
    count = 0
    while True:
        frame[:] = (49, 52, 49)
        global x_out_2, y_out_2

        cvui.text(frame, 10, 10, "Press ESC to exit.")
        cvui.text(frame, 10, 25, "Press 'Save' button than")
        cvui.text(frame, 10, 40, "you need to press'next' button")
        cvui.text(frame, 10, 55, "again to the next photo.")
        cvui.text(frame, 10, 80, f"No. {photo+1} / {len(img_file_list)} ; {hit_x}, {hit_y}")
        cvui.text(frame, 10, 95, f"Saved Count: {count}")
        
        if cvui.button(frame, 10, 120, "Click to save"):
            name=''.join(random.choices(string.ascii_letters+string.digits+"-", k=36))
            if hit_x == get_photo(img_file_list[photo])[0] and hit_y == get_photo(img_file_list[photo])[1]:
                cv2.imwrite(path_of_new + '/' + str(get_photo(img_file_list[photo])[0]) + "_" + str(get_photo(img_file_list[photo])[1]) + '_' + str(name) + ".jpg", img)
            else:
                cv2.imwrite(path_of_new + '/' + str(hit_x) + '_' + str(hit_y) + '_' + str(name) + '.jpg', img)
            print("Image saved")
            print(f"Image saved as {path_of_new + '/' + str(hit_x) + '_' + str(hit_y) + '_' + str(name) + '.jpg'}")
            photo += 1 if photo < len(img_file_list) - 1 else 0
            if photo < len(img_file_list):
                start_image()
                hit_point_reset()
                torch_model_predict(img_file_list[photo])
            else:
                cvui.button(frame, 10, 120, "Maxium photo!")
            count += 1

        if cvui.button(frame, 10, 150, "Click to back photo"):
            photo -= 1 if photo > 0 else 0
            # hit_x, hit_y = get_photo(img_file_list[photo])
            start_image()
            hit_point_reset()
            if model_bool_auto == True:
                torch_model_predict(img_file_list[photo])
            x_out_2, y_out_2 = -1, -1
            
        if cvui.button(frame, 10, 180, "Click to next photo"):
            photo += 1 if photo < len(img_file_list) - 1 else 0
            # hit_x, hit_y = get_photo(img_file_list[photo])
            start_image()
            hit_point_reset()
            if model_bool_auto == True:
                torch_model_predict(img_file_list[photo])
            x_out_2, y_out_2 = -1, -1

        if cvui.button(frame, 125, 120, "Click to Exit"):
            break
        if cvui.button(frame, 10, 210, button_name):
            if model == None:
                print("model loading")
                button_name = "Loading model"
                model_load, _ = QFileDialog.getOpenFileName(None, "Open File", "", "PTH Files (*.pth)")
                torch_model_loader(model_load)
                button_name = "Model auto prediction ON"
            else:
                if model_bool_auto == True:
                    model_bool_auto = False
                    button_name = "Model auto prediction OFF"
                # torch_model_predict(img_file_list[photo])
                elif model_bool_auto == False:
                    model_bool_auto = True
                    button_name = "Model auto prediction ON"
                
        if cvui.button(frame, 165, 240, "predict again"):
            if model != None:
                torch_model_predict(img_file_list[photo])
            else:
                print("model not loaded")

        if cvui.button(frame, 10, 240, "Click to reset model") and model != None:
            model = None
            button_name = "Load Model"
            
        # cvui.image(frame, 0, 0, img2)
        cv2.imshow(WINDOW_NAME+"original", img1)

        cvui.update()
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) == 27:
            break
        if cv2.waitKey(1) == ord(" "):
            torch_model_predict(img_file_list[photo])

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
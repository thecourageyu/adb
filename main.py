# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Installation for pytesseract
# Prerequisites
# 1. pip install python-imaging
# 2. install tesseract
#   - https://github.com/tesseract-ocr/tesseract
#   - https://tesseract-ocr.github.io/tessdoc/Installation.html
#   - https://github.com/UB-Mannheim/tesseract/wiki
# 3. set tesseract.exe (C:\Program Files\Tesseract-OCR) to environment variable PATH

# Install the Pillow and pytesseract packages.
# pip install pillow
# pip install pytesseract
#   - https://pypi.org/project/pytesseract/
# pip install tox
# tox (test installation is ok)

import logging
import os
import subprocess
import re
import time

import matplotlib.pyplot as plt


import cv2
import numpy as np


import sys

from PIL import Image
# import pytesseract

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# # fn = 'E:/OneDrive/OneDrive - Foxconn/Project/adb/rescaled_img_1.png'
# fn = 'E:/OneDrive/OneDrive - Foxconn/Project/adb/img5.jpg'
# img = cv2.imread(fn)
# text = pytesseract.image_to_string(img)
# print(type(text), len(text), text.strip())

# config = {"emulater": 127.0.0.1:, }

# +
bright_red_color = [255, 9, 0]  # color of end point for hp bar
bright_red_color = [255, 90, 70]  # color of end point for hp bar

bright_bule_color = [7, 255, 255]  # color of end point for mp bar 

# +
port = "5556"
emulator = "127.0.0.1:{}".format(port)

screen_resolution = [1600, 900]

hp_bar_rec = {"y": [27, 40], "x": [93, 346]}
mp_bar_rec = {"y": [47, 60], "x": [85, 340]}

# hp_x1, hp_y1, hp_x2, hp_y2 = [89, 14, 355, 40]
hp_x1, hp_y1, hp_x2, hp_y2 = [90, 10, 208, 40]
fhp_x1, fhp_y1, fhp_x2, fhp_y2 = [220, 10, 300, 40]

# mp_x1, mp_y1, mp_x2, mp_y2 = [85, 43, 350, 65]
# mp_x1, mp_y1, mp_x2, mp_y2 = [90, 40, 205, 65]
mp_x1, mp_y1, mp_x2, mp_y2 = [130, 40, 205, 60]

# fmp_x1, fmp_y1, fmp_x2, fmp_y2 = [220, 40, 300, 65]
fmp_x1, fmp_y1, fmp_x2, fmp_y2 = [217, 40, 270, 65]

button1 = [610, 800]
button4 = [882, 800]
button5 = [1454, 800]
button6 = [1167, 800]
button7 = [1265, 800]
button8 = [1360, 800]
button9 = [1440, 800]
button10 = [1535, 800]

config = {"emulator": emulator, 
          "screen_resolution": screen_resolution,
          "hp_bar_rec": hp_bar_rec,
          "mp_bar_rec": mp_bar_rec,
          "hp_x1": hp_x1, "hp_y1": hp_y1, "hp_x2": hp_x2, "hp_y2": hp_y2, 
          "fhp_x1": fhp_x1, "fhp_y1": fhp_y1, "fhp_x2": fhp_x2, "fhp_y2": fhp_y2, 
          "mp_x1": mp_x1, "mp_y1": mp_y1, "mp_x2": mp_x2, "mp_y2": mp_y2,
          "fmp_x1": fmp_x1, "fmp_y1": fmp_y1, "fmp_x2": fmp_x2, "fmp_y2": fmp_y2, 
          "btn": {"btn1": button1, "btn4": button4, "btn5": button5, "btn7": button7, "btn8": button8, "btn9": button9, "btn10": button10}}


# -

def show_current_states(img, current_hp, full_hp, current_mp, full_mp):
    fig = plt.figure(figsize=(16, 16))
    ax1 = plt.subplot2grid((6, 4), (0, 0), rowspan=4, colspan=4)
    ax1.imshow(img)
    ax1.axis("off")
    ax2 = plt.subplot2grid((6, 4), (4, 0), rowspan=1, colspan=2)
    ax2.imshow(current_hp)
    ax2.axis("off")
    ax21 = plt.subplot2grid((6, 4), (4, 2), rowspan=1, colspan=2)
    ax21.imshow(full_hp)
    ax21.axis("off")
    ax3 = plt.subplot2grid((6, 4), (5, 0), rowspan=1, colspan=2)
    ax3.imshow(current_mp)
    ax3.axis("off")
    ax31 = plt.subplot2grid((6, 4), (5, 2), rowspan=1, colspan=2)
    ax31.imshow(full_mp)
    ax31.axis("off")
    # plt.axis('off')
    plt.tight_layout()
    plt.show()


class CurrentState:
    def __init__(self, img, x1, y1, x2, y2):

        self.state = img[y1:y2, x1:x2]

# +
def screenshot(config, bgr2gray: bool = False):
    emulator = config["emulator"]
    pipe = subprocess.Popen("adb {} shell screencap -p".format("" if emulator is None else "-s {}".format(emulator)),
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            shell=True)


    image_bytes = pipe.stdout.read().replace(b'\r\n', b'\n')

    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    if bgr2gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ret, image = cv2.threshold(image, 160, 250, cv2.THRESH_BINARY)

    current_hp = image[config["hp_y1"]:config["hp_y2"], config["hp_x1"]:config["hp_x2"]]
    current_mp = image[config["mp_y1"]:config["mp_y2"], config["mp_x1"]:config["mp_x2"]]    
    full_hp = image[config["fhp_y1"]:config["fhp_y2"], config["fhp_x1"]:config["fhp_x2"]]
    full_mp = image[config["fmp_y1"]:config["fmp_y2"], config["fmp_x1"]:config["fmp_x2"]]    
    
#     arr1 = np.zeros((current_hp.shape[1], current_hp.shape[1]))
#     arr1.fill(np.mean(current_hp))
#     arr2 = np.zeros((current_mp.shape[1], current_mp.shape[1]))
#     arr2.fill(np.mean(current_mp))
#     arr1[100:126, :] = current_hp
#     arr2[100:124, :] = current_mp
    
#     arr1 = np.ndarray((int(current_mp.shape[1] / 2), current_mp.shape[1]))
#     arr1.fill(np.mean(current_mp))
#     current_mp = np.concatenate([arr1, current_mp, arr1], axis=0)
    
    return [image, current_hp, full_hp, current_mp, full_mp]
#     return [image, arr1, arr2]


def bright_red_similarity(x):
    return color_similarity(x, bright_red_color)


def bright_blue_similarity(x):
    return color_similarity(x, bright_bule_color)


def color_similarity(x, y):
    if np.all(x <= 1e-7):
        x = np.add(x, 1e-7)
        
    if euclidean_norm(x) <= 1e-7 or euclidean_norm(y) <= 0.05:
        print(np.sum(np.multiply(x, y)), euclidean_norm(x), euclidean_norm(y), x,  np.sqrt(np.sum(np.power(x, 2))))
    
    return np.sum(np.multiply(x, y)) / (euclidean_norm(x) * euclidean_norm(y))


def euclidean_norm(x):
    return np.sqrt(np.sum(np.power(x, 2)))


# +
def get_hp_percentile(img: np.ndarray, threshold_for_similarity: float = 99):

    img = img.astype(np.float64)
    bright_color = bright_red_similarity
        
    similarity = np.mean(
        np.apply_along_axis(bright_color, axis=2, arr=img), # shape = (width, length)
        axis=0) # shape = (length, )

    length = len(similarity)
    threshold = np.percentile(similarity, threshold_for_similarity)
    most_similar = np.where(similarity > threshold)[0]
    hp_percentile = most_similar / length
    if len(hp_percentile) == 0:
        return -1
    else:
        return hp_percentile[-1]
    

def get_mp_percentile(img: np.ndarray, threshold_for_similarity: float = 99):

    img = img.astype(np.float64)
    bright_color = bright_blue_similarity
        
    similarity = np.mean(
        np.apply_along_axis(bright_color, axis=2, arr=img), # shape = (width, length)
        axis=0) # shape = (length, )

    length = len(similarity)
    threshold = np.percentile(similarity, threshold_for_similarity)
    most_similar = np.where(similarity > threshold)[0]
    mp_percentile = most_similar / length
    if len(mp_percentile) == 0:
        return -1
    else:
        return mp_percentile[-1]
   


# -

def check_rgb_changes(img: np.ndarray, threshold_for_changes: float = 95, return_positions: bool = False):
    """
    img (np.ndarray): rgb image with shape = (width (y), length (x), 3 (channel))
    """
    
    # difference of aixs=1, arr[:, i, :] = img[:, i + 1, :] - img[:, i, :]
    length_diff_on_channel = np.diff(img, axis=1)  # shape = (width, length - 1, 3)
    length_diff = np.mean(
        np.apply_along_axis(euclidean_norm, axis=2, arr=length_diff_on_channel), # shape = (width, length - 1)
        axis=0)  # shape = (length - 1, )
    
    n = len(length_diff)
    
    if return_positions:
        rgb_changes = np.arange(n)[length_diff >= np.percentile(length_diff, threshold_for_changes)]  # shape = (length - 1, ) 
    else:  
        rgb_changes = np.arange(n)[length_diff >= np.percentile(length_diff, threshold_for_changes)] / n  # shape = (length - 1, ) 
    return rgb_changes


# +
# activation_condition = {"btn1": ["mp", ">=" 0.8, 1], "btn8": ["hp", "<=" 0.5, 0], "btn9": ["hp", "<=" 0.85, 0]}


# current_state = {"hp": {"prev": -999, "prev_full": -999, "curr": -999, "curr_full": -999, "ratio": -999}, 
#                  "mp": {"prev": -999, "prev_full": -999, "curr": -999, "curr_full": -999, "ratio": -999}}


# for p_type in ["hp", "mp"]:        
#     if p_type == "hp":
#         current_p = current_hp
#     else:
#         current_p = current_mp

#     current_state[p_type].update({"curr": current_state[p_type]["prev"]})
#     current_state[p_type]["curr_full"] = current_state[p_type]["prev_full"]
        
#     re_search = re.search("(.*)/(.*)\n", current_p)
#     if re_search:
#         grp = re_search.groups()
#         if len(grp) == 2:
#             current_state[p_type].update({"curr":  grp[0], "prev": grp[0]})
#             current_state[p_type].update({"curr_full":  grp[1], "prev_full": grp[1]})

#     ratio = (current_state[p_type]["curr_full"] - current_state[p_type]["curr"]) / current_state[p_type]["curr_full"]
#     current_state[p_type].update({"ratio": ratio})
    
# # max_wait 
# # for 
# # for n_wait in 
# for k, v in activation_condition:  # k: btn, v: [p_type, >, >=, < or <=, ratio, wait seconds]

#     if eval("{} {} {}".format(v[0], v[1], current_state[v[0]]["ratio"])):
#         x, y = config["btn"][k]
#         subprocess.check_output("adb shell input tap {} {}".format(x, y), shell=True)

#     print(curr_hp, full_hp)
# -
5 / 0.45 / 0.37  # +6

5 / 0.45 / 0.37 / 0.11 # +7

7 / 0.2 # +7

7 / 0.2 / 0.1  # +8

 # +8


# +
def action(config,
           logger,
           prev_hp_prectile: float = None,
           prev_mp_prectile: float = None,
           activation_condition: dict = {#"btn4": ["mp", "<=", 0.75, 1],
                                         "btn8": ["hp", "<=", 0.5, 0],
                                         "btn9": ["hp", "<=", 0.85, 0],
                                         # "btn10": ["hp", "<=", 0.85, 0],
           }):

    emulator = config["emulator"]

    img, current_hp, full_hp, current_mp, full_mp = screenshot(config, bgr2gray=False)

    rec = config["hp_bar_rec"]
    hp_bar = img[rec["y"][0]:rec["y"][1], rec["x"][0]:rec["x"][1]]
    current_hp_percentile = get_hp_percentile(hp_bar)
    
    rec = config["mp_bar_rec"]
    mp_bar = img[rec["y"][0]:rec["y"][1], rec["x"][0]:rec["x"][1], :]
    current_mp_percentile = get_mp_percentile(mp_bar)
    
    if current_hp_percentile == -1 or current_mp_percentile == -1:
        return [-1, -1]
    
#     current_hp = pytesseract.image_to_string(current_hp).strip()
#     full_hp = pytesseract.image_to_string(full_hp).strip()
#     current_mp = pytesseract.image_to_string(current_mp).strip()
#     full_mp = pytesseract.image_to_string(full_mp).strip()

    current_state = {"hp": {"prev": prev_hp_prectile, "prev_full": -999, "curr": -999, "curr_full": -999, "ratio": -999}, 
                     "mp": {"prev": prev_mp_prectile, "prev_full": -999, "curr": -999, "curr_full": -999, "ratio": -999}}

    for p_type in ["hp", "mp"]:        
        if p_type == "hp":
            current_p = current_hp_percentile
#             current_p = current_hp
#             full_p = full_hp
        else:
            current_p = current_mp_percentile
#             current_p = current_mp
#             full_p = full_mp

        current_state[p_type].update({"curr": current_p})

        if current_state[p_type]["prev"] is not None:
            print( current_state[p_type]["prev"], current_p, "<<<<<<<<<<<<<<<<<<<<<<365")
            if abs(current_state[p_type]["prev"] - current_p) > 0.2:
                current_state[p_type].update({"curr": current_state[p_type]["prev"]})
                logger.error(">>> estimation failed {}! use previous point {}!\n".format(current_p, current_state[p_type]["prev"]))
                
            
        logger.info(">>> current {}: {} \n".format(p_type, current_state[p_type]["curr"]))

#         current_state[p_type].update({"curr": current_state[p_type]["prev"]})
#         current_state[p_type]["curr_full"] = current_state[p_type]["prev_full"]

#         try:
#             current_state[p_type].update({"curr": float(current_p), "prev": float(current_p)})
#             current_state[p_type].update({"curr_full": float(full_p), "prev_full": float(full_p)})
#             logger.info(">>> current {}: {} / {}\n".format(p_type, current_state[p_type]["curr"], current_state[p_type]["curr_full"]))
#
#         except Exception as err_msg:
#             logger.info(">>> ocr failed! got error: {}\nuse previous point!\n".format(err_msg))

        # ratio = (current_state[p_type]["curr_full"] - current_state[p_type]["curr"]) / current_state[p_type]["curr_full"]
#         ratio = abs(current_state[p_type]["curr"]) / abs(current_state[p_type]["curr_full"])

#         current_state[p_type].update({"ratio": ratio})


    for k, v in activation_condition.items():  # k: btn, v: [p_type, >, >=, < or <=, ratio, wait seconds]

#         if eval("{} {} {}".format(current_state[v[0]]["ratio"], v[1], v[2])):

        if eval("{} {} {}".format(current_state[v[0]]["curr"], v[1], v[2])):
            x, y = config["btn"][k]
            # "adb {} shell screencap -p".format("" if emulator is None else "-s {}".format(emulator)
            subprocess.check_output("adb {} shell input tap {} {}".format("" if emulator is None else "-s {}".format(emulator),
                                                                          x, y), shell=True)
            time.sleep(v[3])
            logger.info("adb {} shell input tap {} {} ({} {} {}, button: {})".format("" if emulator is None else "-s {}".format(emulator),
                                                                                     x, y, current_state[v[0]]["ratio"], v[1], v[2], k))

#         print(curr_hp, full_hp)

    return [current_hp_percentile, current_mp_percentile]


# +
# img = cv2.imread('./screencap.png')
# -

def scan_screenshot(prepared):
    screenshot = cv2.imread('./screencap.png')
    result = cv2.matchTemplate(screenshot, prepared, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return {'screenshot': screenshot, 'min_val': min_val, 'max_val': max_val, 'min_loc': min_loc, 'max_loc': max_loc}


def scan_screenshot(prepared):
    _screenshot = screenshot()
    result = cv2.matchTemplate(_screenshot, prepared, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return {'screenshot': _screenshot, 'min_val': min_val, 'max_val': max_val, 'min_loc': min_loc, 'max_loc': max_loc}


def calculated(result, shape):
    mat_top, mat_left = result['max_loc']
    prepared_height, prepared_width, prepared_channels = shape

    x = {
        'left': int(mat_top),
        'center': int((mat_top + mat_top + prepared_width) / 2),
        'right': int(mat_top + prepared_width),
    }

    y = {
        'top': int(mat_left),
        'center': int((mat_left + mat_left + prepared_height) / 2),
        'bottom': int(mat_left + prepared_height),
    }

    return {
        'x': x,
        'y': y,
    }

# Press the green button in the gutter to run the script.



if __name__ == '__main__':


    config["emulator"] = "127.0.0.1:5556"
    # +
    # arr1 = np.ndarray((int(current_mp.shape[1] / 2), current_mp.shape[1]))
    # arr1.fill(np.mean(current_mp))
    # np.concatenate([arr1, current_mp, arr1], axis=0).shape, arr1.shape, current_mp.shape
    # -

    img, current_hp, current_mp, full_hp, full_mp = screenshot(config, bgr2gray=False)
    print(img.shape, current_hp.shape, current_mp.shape)
    show_current_states(img, current_hp, full_hp, current_mp, full_mp)

    hp_bar = img[27:40, 93:346]
    print(hp_bar.shape)
    plt.imshow(hp_bar)

    340 - 85 + 1, 347 - 93 + 1, 40 - 27 + 1, 60 - 47 + 1

    mp_bar = img[47:60, 85:340]
    plt.imshow(mp_bar)

    get_hp_percentile(hp_bar)

    get_mp_percentile(mp_bar)

    check_rgb_changes(hp_bar)[-1]

    hp_bar[0, 0, :]

    np.where(hp_bar[0, 0, :] == 31)[0] / 10

    logger = logging.getLogger("__name__")
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    print('adb connect {}'.format(config["emulator"]), "<<<<<<<<<<<<<<<<<<<<<<")
    # subprocess.check_output('adb connect {}'.format(config["emulator"]), shell=True)
    # os.system('adb connect {}'.format(config["emulator"]))
    print('adb connect {}'.format(config["emulator"]))
    prev_hp_prectile = None
    prev_mp_prectile = None
    while True:

        prev_hp_prectile, prev_mp_prectile = action(config, logger, prev_hp_prectile=prev_hp_prectile, prev_mp_prectile=prev_mp_prectile)
        print(prev_hp_prectile, prev_mp_prectile)
        if prev_hp_prectile == -1:
            prev_hp_prectile = None
            
        if prev_mp_prectile == -1:
            prev_mp_prectile = None
    #     # 不斷刷新模擬器截圖
    #     screenshot()
    #
    #     # 範例一、判斷目標物件並執行點擊
    #     # 先從圖庫當中，找出你想偵測的圖片
    #     target = cv2.imread('./images/XXX.png')
    #     # 丟去跟畫面做比對
    #     result = scan_screenshot(target, screen)
    #     # 判斷畫面是否有跟圖片相符
    #     if result['max_val'] > 0.9999:
    #         # 對模擬器按圖片的中心點位置
    #         points = calculated(result, target.shape)
    #         subprocess.check_output('adb shell input tap %d %d' % (x, y), shell=True)



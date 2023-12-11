import win32api, win32con, win32gui
from PIL import Image, ImageGrab
import torch
import torchvision


def get_window(window_name):
    handle = win32gui.FindWindow(0, window_name)
    # 获取窗口句柄
    if handle == 0:
        return None
    else:
        # 返回坐标值和handle
        return win32gui.GetWindowRect(handle), handle


def fetch_image(window_name):
    (x1, y1, x2, y2), handle = get_window(window_name)
    # 发送还原最小化窗口的信息
    win32gui.SendMessage(handle, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0)
    # 设为高亮
    win32gui.SetForegroundWindow(handle)
    # 截图
    grab_image = ImageGrab.grab((x1, y1, x2, y2))
    return grab_image


if __name__ == '__main__':
    name = 'Vampire Survivors'
    img = fetch_image(name)
    # img.save("test.png")
    trans = torchvision.transforms.ToTensor()
    t = trans(img)
    print(t.shape)
    print(t[2][422][466])

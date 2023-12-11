import pyautogui as pg
import win32api, win32con, win32gui
from PIL import Image, ImageGrab


def get_window(window_name):
    handle = win32gui.FindWindow(0, window_name)
    # 获取窗口句柄
    if handle == 0:
        return None
    else:
        # 返回坐标值和handle
        return win32gui.GetWindowRect(handle), handle


if __name__ == '__main__':
    # (x1, y1, x2, y2), handle = get_window("Vampire Survivors")
    # # 发送还原最小化窗口的信息
    # win32gui.SendMessage(handle, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0)
    # # 设为高亮
    # win32gui.SetForegroundWindow(handle)
    # pg.PAUSE = .01
    # pg.press('esc')
    # pg.PAUSE = 1
    # pg.keyDown('a')
    # pg.PAUSE = .01
    # pg.press('a')
    # pg.press('esc')
    (x1, y1, x2, y2), handle = get_window("Vampire Survivors")
    # 发送还原最小化窗口的信息
    win32gui.SendMessage(handle, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0)
    # 设为高亮
    win32gui.SetForegroundWindow(handle)
    pg.PAUSE = 1.5
    pg.press('space')
    pg.press('space')
    pg.press('space')
    pg.press('d')
    pg.press('space')
    pg.PAUSE = .01
    pg.press('esc')

import pyMeow
import win32api, win32con, win32gui
from PIL import Image, ImageGrab
import torch
import torchvision
import pyautogui as pg


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


class VSEnv:
    def __init__(self):
        self.__last_exp = 0
        self.__last_hp = 0

        self.__img2tensor = torchvision.transforms.ToTensor()

        self.__process = pyMeow.open_process("VampireSurvivors.exe")
        self.__module = pyMeow.get_module(self.__process, "GameAssembly.dll")
        self.__addr_base = self.__module["base"] + 0x04291000
        offset = [0xB8, 0x18, 0x20, 0x18, 0x38, 0x38, 0x168]
        self.__addr_hp = pyMeow.pointer_chain(self.__process, self.__addr_base, offset)
        offset = [0xB8, 0x18, 0x20, 0x18, 0x38, 0x38, 0x168]
        self.__addr_exp = pyMeow.pointer_chain(self.__process, self.__addr_base, offset)

    def __get_screen(self, name="Vampire Survivors"):
        img = fetch_image(name)
        ret = self.__img2tensor(img)
        return ret

    def __get_state(self):
        screen = self.__get_screen()
        hp = self.read_hp()
        exp = self.read_exp()
        return [screen, hp, exp]

    def __get_reward(self):
        # Rewards in stage 1, only calculate the difference of hp
        # get minus rewards when lose hp and positive rewards when get heal
        reward = self.read_hp() - self.__last_hp
        return reward

    def read_hp(self):
        return pyMeow.r_int(self.__process, self.__addr_hp)

    def read_exp(self):
        return pyMeow.r_int(self.__process, self.__addr_exp)

    def reset(self):
        pg.PAUSE = 1.5
        pg.press('space')
        pg.press('space')
        pg.press('space')
        pg.press('d')
        pg.press('space')
        state = self.__get_state()
        pg.PAUSE = .01
        pg.press('esc')
        reward = self.__get_reward()
        done = False
        if self.read_hp() <= 0:
            done = True
        return state, reward, done

    def step(self, action, step_time):
        pg.PAUSE = .01
        pg.press('esc')
        pg.PAUSE = step_time
        if action is 0:
            pg.keyDown('w')
            pg.PAUSE = .01
            pg.press('w')
        elif action is 1:
            pg.keyDown('w', 'd')
            pg.PAUSE = .01
            pg.press('w', 'd')
        elif action is 2:
            pg.keyDown('d')
            pg.PAUSE = .01
            pg.press('d')
        elif action is 3:
            pg.keyDown('s', 'd')
            pg.PAUSE = .01
            pg.press('s', 'd')
        elif action is 4:
            pg.keyDown('s')
            pg.PAUSE = .01
            pg.press('s')
        elif action is 5:
            pg.keyDown('s', 'a')
            pg.PAUSE = .01
            pg.press('s', 'a')
        elif action is 6:
            pg.keyDown('a')
            pg.PAUSE = .01
            pg.press('a')
        elif action is 7:
            pg.keyDown('a', 'w')
            pg.PAUSE = .01
            pg.press('a', 'w')
        elif action is 8:
            pg.PAUSE = .01
            pg.press('space')
        state = self.__get_state()
        pg.press('esc')
        reward = self.__get_reward()
        done = False
        if self.read_hp() <= 0:
            done = True
        return state, reward, done

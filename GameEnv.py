import struct
import pyMeow
import win32api, win32con, win32gui
from PIL import Image, ImageGrab
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


def bytes2float(byte):
    if byte == 0x0:
        return 0.
    temp = hex(byte)
    ba = bytearray()
    ba.append(int(temp[2:4], 16))
    ba.append(int(temp[4:6], 16))
    ba.append(int(temp[6:8], 16))
    ba.append(int(temp[8:10], 16))
    return struct.unpack("!f", ba)[0]


class VSEnv:
    def __init__(self):
        self.__last_exp = 0.
        self.__last_hp = 100.

        self.__img2tensor = torchvision.transforms.ToTensor()

        self.__process = pyMeow.open_process("VampireSurvivors.exe")
        self.__module = pyMeow.get_module(self.__process, "GameAssembly.dll")
        addr_character_controller = self.__module["base"] + 0x042BE128
        offset = [0xC0, 0x1D8, 0x98, 0x0, 0x18, 0x28, 0x168]
        self.__addr_hp = pyMeow.pointer_chain(self.__process, addr_character_controller, offset)
        offset = [0xC0, 0x1D8, 0x98, 0x0, 0x18, 0x28, 0x17C]
        self.__addr_exp = pyMeow.pointer_chain(self.__process, addr_character_controller, offset)
        offset = [0x70, 0x58, 0x64]
        self.__addr_coins = pyMeow.pointer_chain(self.__process, addr_character_controller, offset)

    def __get_screen(self, name="Vampire Survivors"):
        img = fetch_image(name)
        ret = self.__img2tensor(img)
        return ret

    def __get_state(self):
        state = screen = self.__get_screen()
        # hp = self.read_hp()
        # exp = self.read_exp()
        # state = [screen, hp, exp]
        return state

    def __get_reward(self):
        # Rewards in stage 1, only calculate the difference of hp
        # get minus rewards when lose hp and positive rewards when get heal
        if self.__last_hp is 0.:
            return 0
        temp = self.read_hp()
        reward = temp - self.__last_hp
        self.__last_hp = temp
        return reward

    def read_hp(self):
        return bytes2float(pyMeow.r_int(self.__process, self.__addr_hp))

    def read_exp(self):
        return bytes2float(pyMeow.r_int(self.__process, self.__addr_exp))

    def read_coins(self):
        return bytes2float(pyMeow.r_int(self.__process, self.__addr_coins))

    def reset(self):
        self.__last_exp = 0.
        self.__last_hp = 0.
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

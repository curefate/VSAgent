import struct

import pyMeow


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


if __name__ == '__main__':
    process = pyMeow.open_process("VampireSurvivors.exe")
    module = pyMeow.get_module(process, "GameAssembly.dll")
    base_addr = module["base"] + 0x042BE128
    print("------------HP-------------")
    offsets = [0xC0, 0x1D8, 0x98, 0x0, 0x18, 0x28, 0x168]
    health_addr = pyMeow.pointer_chain(process, base_addr, offsets)
    health_value = pyMeow.r_int(process, health_addr)
    print(health_value)
    print(hex(health_value))
    print(bytes2float(health_value))
    print("------------EXP------------")
    offsets = [0xC0, 0x1D8, 0x98, 0x0, 0x18, 0x28, 0x17C]
    exp_addr = pyMeow.pointer_chain(process, base_addr, offsets)
    exp_value = pyMeow.r_int(process, exp_addr)
    print(exp_value)
    print(hex(exp_value))
    print(bytes2float(exp_value))
    print("-----------TIME------------")
    offsets = [0xB8, 0x0, 0x10, 0x28, 0x100, 0x28, 0x2A0]
    time_addr = module["base"] + 0x0429FC28
    time_addr = pyMeow.pointer_chain(process, time_addr, offsets)
    time_value = pyMeow.r_int(process, time_addr)
    print(time_value)
    print(hex(time_value))
    print(bytes2float(time_value))
    while True:
        print("------------HP-------------")
        health_value = pyMeow.r_int(process, health_addr)
        print(health_value)
        print(hex(health_value))
        print(bytes2float(health_value))
        if bytes2float(health_value) == 0:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

import serial

arduino = serial.Serial('/dev/cu.usbmodem1411', 9600)
prev = arduino.read()
while(True):
    buttonIn = arduino.read()
    if buttonIn != prev:
        print("{0:b}".format(int.from_bytes(arduino.read(), byteorder='big')))
        prev = buttonIn

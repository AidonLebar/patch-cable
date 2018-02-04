import serial

arduino = serial.Serial('/dev/cu.usbmodem1411', 9600)
prev = int.from_bytes(arduino.read(), byteorder = 'big')
prevPotval = int(arduino.readline())
while(True):
    controlIn = int.from_bytes(arduino.read(), byteorder = 'big')
    potIn = int(arduino.readline())
    if controlIn != prev:
        if controlIn & 0b00000001 != 0 and prev & 0b00000001 == 0:
            print("button 7.")

        if controlIn & 0b00000010 != 0 and prev & 0b00000010 == 0:
            print("button 6.")

        if controlIn & 0b00000100 != 0 and prev & 0b00000100 == 0:
            print("button 5.")

        if controlIn & 0b00001000 != 0 and prev & 0b00001000 == 0:
            print("button 4.")

        if controlIn & 0b00010000 != 0 and prev & 0b00010000 == 0:
            print("button 3.")

        if controlIn & 0b00100000 != 0 and prev & 0b00100000 == 0:
            print("button 2.")

        if controlIn & 0b01000000 != 0 and prev & 0b01000000 == 0:
            print("button 1.")


        if controlIn & 0b00000001 == 0 and prev & 0b00000001 != 0:
            print("button 7 released.")

        if controlIn & 0b00000010 == 0 and prev & 0b00000010 != 0:
            print("button 6 released.")

        if controlIn & 0b00000100 == 0 and prev & 0b00000100 != 0:
            print("button 5 released.")

        if controlIn & 0b00001000 == 0 and prev & 0b00001000 != 0:
            print("button 4 released.")

        if controlIn & 0b00010000 == 0 and prev & 0b00010000 != 0:
            print("button 3 released.")

        if controlIn & 0b00100000 == 0 and prev & 0b00100000 != 0:
            print("button 2 released.")

        if controlIn & 0b01000000 == 0 and prev & 0b01000000 != 0:
            print("button 1 released.")

    if prevPotval != potIn:
        print("potentiometer: {}".format(1/(potIn+1)))

    prev = controlIn
    prevPotval = potIn

import serial

def inputMonitor(input):
    arduino = serial.Serial('/dev/cu.usbmodem1411', 9600)
    arduino.reset_input_buffer()
    prev = int.from_bytes(arduino.read(), byteorder = 'big')
    prevPotval = int(arduino.readline())
    while(True):
        controlIn = int.from_bytes(arduino.read(), byteorder = 'big')
        potIn = int(arduino.readline())
        if controlIn != prev:
            if controlIn & 0b00000001 != 0 and prev & 0b00000001 == 0:
                print("button 7.")
                input[6] = 1.0

            if controlIn & 0b00000010 != 0 and prev & 0b00000010 == 0:
                print("button 6.")
                input[5] = 1.0

            if controlIn & 0b00000100 != 0 and prev & 0b00000100 == 0:
                print("button 5.")
                input[4] = 1.0

            if controlIn & 0b00001000 != 0 and prev & 0b00001000 == 0:
                print("button 4.")
                input[3] = 1.0

            if controlIn & 0b00010000 != 0 and prev & 0b00010000 == 0:
                print("button 3.")
                input[2] = 1.0

            if controlIn & 0b00100000 != 0 and prev & 0b00100000 == 0:
                print("button 2.")
                input[1] = 1.0

            if controlIn & 0b01000000 != 0 and prev & 0b01000000 == 0:
                print("button 1.")
                input[0] = 1.0


            if controlIn & 0b00000001 == 0 and prev & 0b00000001 != 0:
                print("button 7 released.")
                input[6] = 0.0

            if controlIn & 0b00000010 == 0 and prev & 0b00000010 != 0:
                print("button 6 released.")
                input[5] = 0.0

            if controlIn & 0b00000100 == 0 and prev & 0b00000100 != 0:
                print("button 5 released.")
                input[4] = 0.0

            if controlIn & 0b00001000 == 0 and prev & 0b00001000 != 0:
                print("button 4 released.")
                input[3] = 0.0

            if controlIn & 0b00010000 == 0 and prev & 0b00010000 != 0:
                print("button 3 released.")
                input[2] = 0.0

            if controlIn & 0b00100000 == 0 and prev & 0b00100000 != 0:
                print("button 2 released.")
                input[1] = 0.0

            if controlIn & 0b01000000 == 0 and prev & 0b01000000 != 0:
                print("button 1 released.")
                input[0] = 0.0

        if prevPotval != potIn:
            print("potentiometer: {}".format(1/(potIn+1)))
            input[7] = 1/(potIn+1)

        prev = controlIn
        prevPotval = potIn

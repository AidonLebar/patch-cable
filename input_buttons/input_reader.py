import serial


def input_monitor(inputs):
    arduino = serial.Serial('/dev/cu.usbmodem1411', 9600)
    arduino.reset_input_buffer()
    prev = 0
    prevPotval = 0
    while True:
        try:
            controlIn = int.from_bytes(arduino.read(), byteorder='big')
            potIn = int(arduino.readline())
        except Exception:
            continue
            
        if controlIn != prev:
            if controlIn & 0b00000001 != 0 and prev & 0b00000001 == 0:
                print("button 7.")
                inputs[6] = 1.0

            if controlIn & 0b00000010 != 0 and prev & 0b00000010 == 0:
                print("button 6.")
                inputs[5] = 1.0

            if controlIn & 0b00000100 != 0 and prev & 0b00000100 == 0:
                print("button 5.")
                inputs[4] = 1.0

            if controlIn & 0b00001000 != 0 and prev & 0b00001000 == 0:
                print("button 4.")
                inputs[3] = 1.0

            if controlIn & 0b00010000 != 0 and prev & 0b00010000 == 0:
                print("button 3.")
                inputs[2] = 1.0

            if controlIn & 0b00100000 != 0 and prev & 0b00100000 == 0:
                print("button 2.")
                inputs[1] = 1.0

            if controlIn & 0b01000000 != 0 and prev & 0b01000000 == 0:
                print("button 1.")
                inputs[0] = 1.0


            if controlIn & 0b00000001 == 0 and prev & 0b00000001 != 0:
                print("button 7 released.")
                inputs[6] = 0.0

            if controlIn & 0b00000010 == 0 and prev & 0b00000010 != 0:
                print("button 6 released.")
                inputs[5] = 0.0

            if controlIn & 0b00000100 == 0 and prev & 0b00000100 != 0:
                print("button 5 released.")
                inputs[4] = 0.0

            if controlIn & 0b00001000 == 0 and prev & 0b00001000 != 0:
                print("button 4 released.")
                inputs[3] = 0.0

            if controlIn & 0b00010000 == 0 and prev & 0b00010000 != 0:
                print("button 3 released.")
                inputs[2] = 0.0

            if controlIn & 0b00100000 == 0 and prev & 0b00100000 != 0:
                print("button 2 released.")
                inputs[1] = 0.0

            if controlIn & 0b01000000 == 0 and prev & 0b01000000 != 0:
                print("button 1 released.")
                inputs[0] = 0.0

        if prevPotval != potIn:
            print("potentiometer: {}".format(1/(potIn+1)))
            inputs[7] = 1 / (potIn + 1)

        prev = controlIn
        prevPotval = potIn

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 10:28:40 2018
@author: Raquel Dueñas Suárez
Capturing ground truth for music detection in audio:
    - Insert the name of the file where you want to save the data
    - Click and keep press the left button when hearing music
    - When finished, press the right button
"""

import sys
from datetime import datetime
from pynput import mouse

# VARIABLES
msg = "Indicate whole file path:"
pressed_path = input(msg)
pressed_moments = []
start = datetime.now()


# AUXILIARY FUNCTIONS
def on_move():  # Mouse moving
    pass


def on_click(x, y, button, pressed):  # Mouse click
    if button == mouse.Button.left:
        print('{0}'.format(
            'Pressed' if pressed else 'Released'))
        if pressed:
            pressed_moments.append('Starting music:' + str(datetime.now() - start))
        else:
            pressed_moments.append('Stopping music:' + str(datetime.now() - start))
    else:
        pressed_moments.append('Ending:' + str(datetime.now() - start))
        return False


def on_scroll():  # Mouse scroll
    pass


# MAIN FUNCTION
def main():
    print("Press right button to quit...")
    try:
        # MOUSE LISTENING
        with mouse.Listener(
                on_move=on_move,
                on_click=on_click,
                on_scroll=on_scroll) as mouseListener:
            mouseListener.join()
        sys.stdin.readline()

    except KeyboardInterrupt:
        pass

    finally:

        # SAVE MOUSE EVENTS
        f = open(pressed_path, "w")
        for p in pressed_moments:
            f.write("%s\n" % p)
        f.close()


if __name__ == "__main__":
    main()

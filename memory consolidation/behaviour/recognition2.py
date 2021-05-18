import os
import time 
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from psychopy import core, visual, gui, data, event, clock
from psychopy.tools.filetools import fromFile, toFile
import time
import numpy as np
import random
import serial
from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock
from psychopy.visual import filters
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
import numpy as np
from ctypes import windll
import os  # handy system and path functions
import sys  # to get file system encoding
from psychopy.hardware import keyboard
import time
from time import sleep
from psychopy import core
import serial


send_trigger = False
if send_trigger: s = serial.Serial('COM11')
clock = core.Clock()

# participant information
# ---------------------------------------------------------------------------------------------
def on_press(key):
    global key_code
    if key == Key.space:
        key_code = '11'
        print('pressed')
    

def quit_and_store():
    dataFile.close()    
    core.quit()    
event.globalKeys.add(key='escape', func=quit_and_store)
event.globalKeys.add(key='q', func=quit_and_store)
        
# __________________________________________________________________
# Get input from participant and prepare data file to store behavioural results
expName = 'VWM.py'
expInfo = {'participant': 'Use the same UNIQUE ID everywhere', 'session': 'dd-mm_x'}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
p_num = expInfo['participant']
sess = expInfo['session']
dataFile = open(f'{p_num}_task_{sess}.csv', 'w')

# key_code similar to other EEG triggers! 11 and 12
dataFile.write('Block,Trial,image_code,key_code,correct_incorrect\n')

# ---------------------------------------------------------------------------------------------
# base_path = "C:\\Users\\Applied Neurotec\\Desktop\\memory_consolidation\\images\\encoding\\"
base_path = "C:\\Users\\alessia\\Documents\\Jobs\\CNT\\CL-amtACS\\memory_consolidation\\"

# experiment starts
# ---------------------------------------------------------------------------------------------
win = visual.Window( allowGUI=True, monitor='testMonitor', fullscr = True,units='deg',color=[1,1,1])
message = 'Now it is time to remember which pictures you saw yesterday. You will see images that were presented to you and new images. If you think you saw the image, press the space bar. If you think you are seeing a new picture do nothing! You saw the picture = space'

instructions = visual.TextStim(win,
                        text=message,bold= True,alignText='center',color=[-1,-1,-1])
instructions.draw()
win.flip()
event.waitKeys()

fixation = visual.GratingStim(win, color=[-1,-1,-1], colorSpace='rgb', tex=None, mask='circle', size=0.2)
message_1 = visual.TextStim(win, pos=[0, +3],color=[-1,-1,-1], colorSpace='rgb',
                            text='Press a key when ready.')

message_1.draw()
fixation.draw()
win.flip()
event.waitKeys()

image_block = ['Block1/','Block2/','Block3/','Block4/','Block5/','Block6/']


for idx in range(len(image_block)):


    path = base_path + image_block[idx]
    #names = [f for f in os.listdir(path) if f.endswith('.jpg')]
    #names = random.sample(names,len(names))
    names = np.arange(20)

    fixation = visual.TextStim(win,text = image_block[idx][:-1],pos=[0, +3],
                        alignText='center',colorSpace='rgb',color=[-1,-1,-1])
    fixation.draw()
    win.flip()
    time.sleep(1)

    message = visual.TextStim(win,text = 'Press a key when ready.',pos=[0, +3],
                        alignText='center',colorSpace='rgb',color=[-1,-1,-1])
    message.draw()
    win.flip()
    event.waitKeys()
    
    for pic, trial in zip(names,range(len(names))):
        global key_code

        key_code = '0'

        # fixation cross presentation
        rand_tim = np.random.uniform(1.2,1.8)
        fixation = visual.TextStim(win,
                                text='+',bold= True, alignText='center',height=10,colorSpace='rgb',color=[-1,-1,-1])
        fixation.draw()
        win.flip()
        time.sleep(rand_tim)

        # image present code
        if send_trigger:
            s.write(str.encode(chr(0)))
            s.write(str.encode(chr(10)))

        # image presentation
        #image_stim = visual.ImageStim(win, image= path  + pic)

        #image_stim.draw()
        win.flip()
        #code = int(pic[4:-4])
        code = 0
        if send_trigger:
            s.write(str.encode(chr(0)))
            s.write(str.encode(chr(code)))
        time.sleep(1)

        #image presentation stop and stimulation stop
        # stops stimulation
        if send_trigger:
            s.write(str.encode(chr(0)))
            s.write(str.encode(chr(1)))

        # question mark presentation
        event.clearEvents()
        fixation = visual.TextStim(win,
                                text='?',bold= True,alignText='center',height=10,colorSpace='rgb',color=[-1,-1,-1])
        fixation.draw()
        win.flip()
        
        pressed = 0
        
        clock.reset()
        time.sleep(1.5)
        keys = event.getKeys(keyList = ["space"])
        if "space" in keys:
            pressed = 1
        event.clearEvents()
        #listener = Listener(on_press=on_press)
        #listener.start()
        #time.sleep(1.5)

        if int(code) < 153 and pressed: 
            key_code = '11'
            response = 1
        elif int(code) > 153 and  pressed:
            response = 0
        else:
            response = 0
        
        #listener.stop()




        dataFile.write('%i, %i, %i, %i, %i\n' %(int(idx)+1, trial, code, int(key_code), response ))

        if send_trigger:
            s.flush()
            s.reset_output_buffer()

    fixation = visual.TextStim(win,text = '',
                        bold= True,alignText='center',height=10,colorSpace='rgb',color=[-1,-1,-1])
    fixation.draw()
    win.flip()
    time.sleep(30)

dataFile.close()
fixation = visual.TextStim(win,
                        text='Thank you for participating',bold= True,alignText='center',color=[-1,-1,-1])
fixation.draw()
win.flip()
time.sleep(1)

    

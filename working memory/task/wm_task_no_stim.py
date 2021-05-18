# ########################################################
# Working Memory Task
# Change detection task: Color of 5 squares 
# Control task: Orientation of grating
# ########################################################

# __________________________________________________________________
# Prepare environment

# Import packages and fucntions
from __future__ import absolute_import, division
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


# __________________________________________________________________
# Change parameters HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Insert correct data folder
os.chdir('C:/Users/Applied Neurotec/Documents/Closed-Loop Working Memory/data')

# Set use_serial = True for EEG recording 
use_serial = True
if use_serial: s = serial.Serial('COM11')

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
dataFile = open(f'{p_num}_no_stim_{sess}.csv', 'w')
dataFile.write('BlockID,Trial,Same,Response,TooSlow,ReactionTime\n')

# __________________________________________________________________
# Set up window, clock and seed
win = visual.Window(
    size=(1920, 1080), fullscr=False, screen=0, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color= 'black', colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
clock = core.Clock()
np.random.seed(42) # Change seed in screening session 

# __________________________________________________________________
# Define help functions

def quit_and_store():
    dataFile.close()    
    core.quit()    
event.globalKeys.add(key='escape', func=quit_and_store)

skip_with_space=visual.TextStim(win=win, text='Press [SPACE] to continue', pos=(0.0, -0.25), height=0.03, color= 'white')

def draw_and_wait(message,win):    
    message.draw()
    skip_with_space.draw(win)
    win.flip()
    event.waitKeys()
    event.clearEvents()
    
def instructions_task(n_blocks):   
    message.text = '''CHANGE DETECTION TASK''' 
    draw_and_wait(message,win)
    message.text = '''5 colored squares will appear\n \n and then quickly disappear'''
    draw_and_wait(message,win)
    message.text = '''After disappearing, another set of colored\n \n squares will be displayed for a short time''' 
    draw_and_wait(message,win)
    message.text = '''Your TASK is to indicate whether the two\n \n sets have the SAME COLOR OR NOT'''
    draw_and_wait(message,win)
    message.text = '''Are they colored the same?\n \n Press 'J' (Ja) for yes \n \n Press 'F' (Falsch) for no'''
    draw_and_wait(message,win)
    message.text = '''For CORRECT choices you WIN %s cents\n \n For WRONG choices you LOSE %s cents''' % (gpr, gpr)
    draw_and_wait(message,win)
    message.text = '''Are the instructions clear?\n\n\n [Y] Proceed [N] Replay'''
    message.draw()
    win.flip()
    key_instr = event.waitKeys(keyList=['y','n'])
    event.clearEvents()
    message.text = '''Press SPACE to begin'''
    draw_and_wait(message,win)
    message.text = 'Block 1/%i' % n_blocks
    message.draw()
    win.flip() 
    sleep(1)
    return

def get_response(b,trial,same,condition,win,message,cross,gpr,use_serial,gain,clock): 
    cross.autoDraw = False
    win.flip()
    wait_time = 0.8
    clock.reset()
    key_res = event.waitKeys(maxWait=wait_time, timeStamped=clock)
    event.clearEvents()
    if not key_res:
        message.text = "TOO SLOW \n \n \n -%s CENT" % gpr
        message.color = 'red'
        response = False
        too_slow = True
        gain -= gpr
        time = wait_time
    else:
        key = key_res[0][0]
        time= key_res[0][1]
        too_slow = False
        if key=='f' and not same or key=='j' and same:
            message.text = "+%s CENT" % gpr
            message.color = 'green'
            response = True
            gain += gpr
            core.wait(wait_time-time/100) 
        else:
            message.text = "-%s CENT" % gpr
            message.color = 'red'
            response = False
            gain -= gpr
            core.wait(wait_time-time/100)
            
    message.draw()    
    win.flip()
    if use_serial: s.write(str.encode(chr(6)))
    core.wait(0.8)
    message.color = 'white'
    
    # Save response data 
    dataFile.write('%i, %i, %i, %i, %i, %i, %.3f\n' %(b, trial, same, condition, response, too_slow, time))
    core.wait(0.8)
    
    return gain

def draw_rectangles(rects,colors,win):
    for i, rect in enumerate(rects):
        rect.fillColor = colors[i]
        rect.lineColor = colors[i]
        rect.draw()
    win.flip()
        
# __________________________________________________________________
# Define objects and parameters

# Define experimental design parameters 
n_blocks = 2
n_trials = 28
n_trials_tot = n_trials * n_blocks
n_rects = 5
gpr = 5 # Gains or losses per round in cents
when_to_stop = 30   # Duration of break

# Define object parameters
colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'brown', 'cyan', 'Aquamarine']
colors_all_trials =  [None] * n_trials_tot
same_all_trials = [None] * n_trials_tot
colors_all_trials_new =  [None] * n_trials_tot
for trial in range(n_trials_tot):
    # Change detection task 
    colors_trial = np.random.choice(colors, n_rects, False) # Colors of target rectangles
    colors_all_trials[trial] = colors_trial
    same = np.random.choice([True, False]) # Change or not
    same_all_trials[trial] = same
    colors_all_trials_new[trial] = colors_all_trials[trial].copy()
    if not same:
        change_i = np.random.choice(n_rects) # Which rectangle should change
        colors_all_trials_new[trial][change_i] = np.random.choice(list(set(colors) - set(colors_trial))) # Change to a color not present before

# Define objects
# Rectangles
rects = [None] * n_rects
radius = 0.28
angle = 2 * np.pi / n_rects
for i in range(n_rects):
    pos = (radius * np.sin(i * angle), radius * np.cos(i * angle))
    rects[i] = visual.Rect(win=win, width=0.2, height=0.2, pos=pos)
# Fixation cross
cross = visual.ShapeStim(win=win, lineColor='gray', vertices=((0, -0.04), (0, 0.04), (0,0), (-0.04,0), (0.04, 0)), closeShape=False)

# Initialize parameters and other sruff
gain = 0
NA = "n/a" #needed for correct annotation of excel spreadsheet during control 

# Generate the order of the phase shifts of the stimulation 
conditions = [1,2,3]
conditions_all_trials = []
for i in np.arange(int(n_trials_tot/len(conditions))):
    conditions_tmp = conditions.copy()
    np.random.shuffle(conditions_tmp)
    conditions_all_trials.append(conditions_tmp)
conditions_all_trials = np.array(conditions_all_trials).flatten()
# __________________________________________________________________
# Start experiment

# Show welcome screen
welcome_text = 'Welcome to our experiment'
message = visual.TextStim(win=win, text=welcome_text, color='white', height=0.07)
draw_and_wait(message, win)

message.text = 'There are enough breaks throughout the experiment\n\n\n Still, you can quit at any time by pressing [ESC]'
message.height = 0.05
draw_and_wait(message, win)

for b in range(n_blocks):

    # __________________________________________________________________
    # Experimental block
               
    # Show instructions only before the first block
    if b == 0:
        instructions_task(n_blocks)
    else:
        message.text = '''CHANGE DETECTION TASK'''
        draw_and_wait(message,win)
        message.text = '''Are the two sets of squares the same?\n \n Press 'J' (Ja) for yes\n \n Press 'F' (Falsch) for no'''
        draw_and_wait(message, win)
        message.text = '''Press SPACE to begin'''
        draw_and_wait(message,win)
        message.text = 'Block %i/%i' % (b+1, n_blocks)
        message.draw()
        win.flip() 
        sleep(1)
        
    # __________________________________________________________________
    # Trial
    
    for trial in range(n_trials):
        condition = conditions_all_trials[trial]
        
        # Fixation cross
        cross.autoDraw = True
        win.flip()
        core.wait(0.4)
        
        # Target rectangles
        draw_rectangles(rects,colors_all_trials[trial],win)
        if use_serial: s.write(str.encode(chr(7)))
        core.wait(0.2)
        
        # Delay
        win.flip()
        if use_serial: s.write(str.encode(chr(condition)))
        core.wait(3.0)
        
        # Probe rectangles
        draw_rectangles(rects,colors_all_trials_new[trial],win)
        if use_serial: s.write(str.encode(chr(1)))
        core.wait(0.2)
        win.flip()
        if use_serial: s.write(str.encode(chr(5)))
        gain = get_response(b,trial,same_all_trials[trial],condition,win,message,cross,gpr,use_serial,gain,clock)
        
    # __________________________________________________________________   
    # Break between blocks
    
    if b < n_blocks-1:
        
        message.text = 'BREAK'
        message.draw()
        win.flip()
        sleep(2)
        
        for w in range(when_to_stop):
            if w < 15:
                rest_time = 30-w
                message.text = rest_time
                message.draw()
                win.flip() 
                sleep(1)
                w = w+1
                event.clearEvents()
            else: 
                if event.getKeys('space'):
                    message.text = 'Skipped Break'
                    message.draw()
                    win.flip() 
                    sleep(1)
                    break
                else:
                    rest_time = 30-w
                    message.text = rest_time
                    message.draw()
                    skip_with_space.draw(win)
                    win.flip()
                    sleep(1)
                    w = w+1
                    
        message.text = 'Get ready!!'
        message.color = 'red'
        message.draw()
        win.flip() 
        sleep(2)
        message.color = 'white'

                
message.text = '''TASK COMPLETE \n \n THANK YOU FOR PLAYING!!'''
draw_and_wait(message, win)

message.text = '''YOU EARNED %s EURO \n \n \n press [SPACE] to exit'''  % round(gain/100,1)
message.draw()
win.flip()
event.waitKeys()
event.clearEvents()
        
dataFile.close()
 
core.quit()
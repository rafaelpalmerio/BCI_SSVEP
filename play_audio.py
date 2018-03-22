# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 19:03:53 2017

@author: Rafael and Lucas
"""

from pygame import mixer # Load the required library
import os

def play_file(freq):
    folder = r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\freq_audio'
    folder = os.path.join(os.getcwd(), 'freq_audio')
    filename = os.path.join(folder, freq+'.mp3')
    mixer.init()
    #mixer.music.load('e:/LOCAL/Betrayer/Metalik Klinik1-Anak Sekolah.mp3')
    mixer.music.load(filename)
    mixer.music.play()
    while mixer.music.get_busy():
        pass
    #mixer.music.stop()
    

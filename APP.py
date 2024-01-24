import os
from tkinter import *

root = Tk()

root.title("Hand Effects App")
root.geometry('600x200')

def run_crystal():
    os.system('crystal_effect.py')


def run_water():
    os.system('water_effect.py')


def run_fire():
    os.system('fire_effect.py')


def run_wind():
    os.system('wind_effect.py')

label = Label(root, text="Choose an effect: ", font="courier 14 normal")
label.grid(row = 0)

button_run_magic = Button(root, text="Crystal", font="courier 14 normal", command=run_crystal)
button_run_magic.grid(row = 2, column= 1, padx = 10, pady = 10)

button_run_fire = Button(root, text="Fire", font="courier 14 normal", command=run_fire)
button_run_fire.grid(row = 2, column= 2, padx = 10, pady = 10)

button_run_wind = Button(root, text="Wind", font="courier 14 normal", command=run_wind)
button_run_wind.grid(row = 2, column= 3, padx = 10, pady = 10)

button_run_water = Button(root, text="Water", font="courier 14 normal", command=run_water)
button_run_water.grid(row = 2, column= 4, padx = 10, pady = 10)

root.mainloop()

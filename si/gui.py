import customtkinter as ctk
from tkinter import *


window = ctk.CTk()
window.geometry('1200x800')
window.title('Shell-Intelligence')

window.iconphoto(True, PhotoImage(file="/home/guna/si/si_term/$!.png"))

def new_tab():
    tab.add('new-tab')
    tab.add('new-tab1')
    tab.pack()

def new_window():
    window2 = ctk.CTkToplevel(window)
    window2.title('New Window')
    label = ctk.CTkLabel(master=window2, text='This is a new window')
    label.pack()

def preferences():
    pass

def exit():
    window.quit()

def zoom_in():
    pass

def zoom_out():
    pass

def zoom_set():
    pass

def split_vertical():
    pass

def split_horizontal():
    pass

def about():
    about_window = ctk.CTkToplevel(window)
    about_window.title('About')
    label = ctk.CTkLabel(master= about_window, text='''Shell-Intelligence is an advanced tool built for making user to easy-interact with operating systems  and command line tools''', height=600, width=600)
    label.pack()

menubar = Menu(window, fg='white', bg='#1a1a1a')
tab = ctk.CTkTabview(master=window, bg_color='#1a1a1a', height=1200, width=1000)

#menu_options  
menu_options = Menu(master=menubar, tearoff=0, bg='#1a1a1a',fg='white')
menu_options.add_command(label= 'new-tab', command =new_tab)
menu_options.add_command(label= 'new window', command =new_window)
menu_options.add_command(label= 'preferences', command =preferences)
menu_options.add_command(label= 'exit', command =exit)
menu_options.add_separator()

#editor_options 
editor_options = Menu(master=menubar, tearoff=0,bg='#1a1a1a',fg='white')
editor_options.add_command(label='Zoom In', command =zoom_in)
editor_options.add_command(label='Zoom Out', command =zoom_out)
editor_options.add_command(label='zoom fix', command =zoom_set)
editor_options.add_separator()

#help menu
help_options = Menu(master=menubar, tearoff=0,bg='#1a1a1a',fg='white')
help_options.add_command(label= 'About', command =about)
help_options.add_separator()

menubar.add_cascade(label= 'Menu', menu= menu_options)
menubar.add_cascade(label='Edit', menu= editor_options)
menubar.add_cascade(label='Help', menu= help_options)

window.configure(menu=menubar)
window.configure(bg='#1a1a1a')


window.mainloop()

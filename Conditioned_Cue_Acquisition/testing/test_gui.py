import os
import sys
import shutil
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import ImageTk, Image

def plot(d1, d2, key):
    PFC_trial = d1[key]
    HPC_trial = d2[key]

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10,4))
    axs[0].set_title(f'\n{key} PFC')
    axs[0].plot(PFC_trial)
    axs[1].set_title(f'{key} HPC')
    axs[1].plot(HPC_trial, c='orange')
    plt.tight_layout()
    plt.savefig(f'temp/{key}.png')
    plt.close()

    # canvas = FigureCanvasTkAgg(fig, master=target_frame)
    # canvas.draw()
    # canvas.get_tk_widget().pack()

def complete_curation():
    curation = [var.get() for var in vars]
    print(curation)
    sys.exit()
    pass

def set_mousewheel(widget, command):
    """Activate / deactivate mousewheel scrolling when
    cursor is over / not over the widget respectively."""
    widget.bind("<Enter>", lambda _: widget.bind_all('<MouseWheel>', command))
    widget.bind("<Leave>", lambda _: widget.unbind_all('<MouseWheel>'))

PFC = np.load('Vehicle_coke_PFC_curated.npy', allow_pickle=True).item()
HPC = np.load('Vehicle_coke_HPC_curated.npy', allow_pickle=True).item()

assert PFC.keys() == HPC.keys()
keys = list(PFC.keys())

# if os.path.exists('temp/') and os.path.isdir('temp/'):
#     shutil.rmtree('temp/')
# os.mkdir('temp')
# for key in keys:
#     plot(PFC,HPC,key)

root = tk.Tk()
root.eval('tk::PlaceWindow . center')
root.title('curation!')
root.geometry('1100x1000')

# main frame to house all components
frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=1)

# canvas to place scrollable element
canvas = tk.Canvas(frame)
canvas.bind_all("<MouseWheel>")
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# scrollbar
scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# config the canvas
canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind(
    '<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all'))
)

# add elements to target frame
target_frame = tk.Frame(canvas)

l1 = tk.Label(target_frame, text='Curation by visual inspection', font=('Helvetica', 25, 'normal'))
l1.pack()

l2 = tk.Label(target_frame, text='Inspect waveforms of trials for artifacts in either region and select the check box beneath each plot to include the trial.\nLeave the check box unchecked otherwise to exclude noisy trials.\nClick "complete curation" to write curated data to disk and exit program.', anchor='e', justify=tk.LEFT)
l2.pack()

# maintaining 100s of plt plots at once was too memory intensive and slowed things down
# intead, i'm writing temp img files for each plot and showing the image
vars = [tk.IntVar() for i in range(len(keys))]
imgs = []
for var, key in zip(vars, keys):
    #c = tk.Checkbutton(target_frame, text='keep plot', variable=var, onvalue=1, offvalue=0, command=plot(PFC, HPC, key))
    #plot(PFC, HPC, key)
    imgs.append(ImageTk.PhotoImage(Image.open(f'temp/{key}.png')))
    panel = tk.Label(target_frame, image=imgs[-1])
    panel.pack()
    c = tk.Checkbutton(target_frame, text='keep plot', variable=var, onvalue=1, offvalue=0)
    c.pack()


button_complete = tk.Button(target_frame, command=complete_curation, text='complete curation', activebackground='blue', activeforeground='white', anchor='center', overrelief='raised', padx=10, pady=5)
button_complete.pack()

canvas.create_window((0,0), window=target_frame, anchor='nw')
root.mainloop()

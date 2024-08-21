import os
import sys
import shutil
import argparse
import tkinter as tk
import numpy as np
from PIL import ImageTk, Image
from tkinter import messagebox

parser = argparse.ArgumentParser(description = 'simultaneously visually curate all trialized ephys data in a gui')
parser.add_argument('PFCpath', metavar='<path>', help='path to dict of PFC data to curate.')
parser.add_argument('HPCpath', metavar='<path>', help='path to dict of HPC data to curate.')
args = parser.parse_args()

class curationGUI:
    def __init__(self, PFCpath, HPCpath):
        self.PFCpath = PFCpath
        self.HPCpath = HPCpath
        self.PFCdata = None
        self.HPCdata = None
        self.keys = None
        self._get_data()

        self.root = tk.Tk()

        #self.root.eval('tk::PlaceWindow . center')
        self.root.title('curation!')
        self.root.geometry('1450x2000')

        # menubar
        self.menubar = tk.Menu(self.root)

        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label='Check all', command=self._checkall)
        self.filemenu.add_command(label='Uncheck all', command=self._uncheckall)
        self.filemenu.add_separator()
        self.filemenu.add_command(label='Help', command=self._help)
        self.filemenu.add_separator()
        self.filemenu.add_command(label='Complete curation', command=self._complete)
        self.filemenu.add_command(label='Exit', command=self._exit)

        self.menubar.add_cascade(menu=self.filemenu, label='File')
        self.root.config(menu=self.menubar)

        # main frame to house all components
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=1)

        # canvas to place scrollable element
        self.canvas = tk.Canvas(self.frame)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        # scrollbar
        self.scrollbar = tk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # config the canvas
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all'))
        )

        # add elements to target frames on scrollable canvas
        # top frame for labels
        self.plotframe = tk.Frame(self.canvas)
        self.plotframe.columnconfigure(0, weight=1)
        self.plotframe.columnconfigure(1, weight=0)
        self.plotframe.columnconfigure(2, weight=1)
        self.plotframe.columnconfigure(3, weight=0)

        # placing each plot and checkbox
        self.checkvars = [tk.IntVar() for i in range(len(self.keys))]
        self.imgs = []
        self.checkboxes = []
        for i, t in enumerate(zip(self.checkvars, self.keys)):
            var, key = t
            self.imgs.append(ImageTk.PhotoImage(Image.open(f'temp/{key}.png')))
            plot = tk.Label(self.plotframe, image=self.imgs[-1])

            j = 0 if i%2 == 0 else 2
            plot.grid(row=i//2, column=j, sticky=tk.W+tk.E)

            c = tk.Checkbutton(self.plotframe, text='keep trial?', variable=var, onvalue=1, offvalue=0)
            c.grid(row=i//2, column=j+1)
            self.checkboxes.append(c)

        # end buttons
        self.chkallbtn = tk.Button(self.plotframe, command=self._checkall, text='Check all', font=('Arial', 16), activebackground='blue', activeforeground='white', anchor='center', overrelief='raised', padx=10, pady=10)
        self.chkallbtn.grid(row=len(self.keys)+1, column=1, padx=5, pady=5, sticky='w')

        self.unchkallbtn = tk.Button(self.plotframe, command=self._uncheckall, text='Uncheck all', font=('Arial', 16), activebackground='blue', activeforeground='white', anchor='center', overrelief='raised', padx=10, pady=10)
        self.unchkallbtn.grid(row=len(self.keys)+2, column=1, padx=5, pady=5, sticky='w')

        self.completebtn = tk.Button(self.plotframe, command=self._complete, text='Complete curation', font=('Arial',16), activebackground='blue', activeforeground='white', anchor='center', overrelief='raised', padx=10, pady=10)
        self.completebtn.grid(row=len(self.keys)+3, column=1, padx=5, pady=5, columnspan=2, sticky='w')

        self.exitbtn = tk.Button(self.plotframe, command=self._exit, text='Exit', font=('Arial',16), activebackground='blue', activeforeground='white', anchor='center', overrelief='raised', padx=10, pady=10)
        self.exitbtn.grid(row=len(self.keys)+4, column=1, padx=5, pady=5, sticky='w')

        self.canvas.create_window((0,0), window=self.plotframe, anchor='nw')
        self.root.mainloop()

    def _help(self):
        messagebox.showinfo(title='Help', message='Help', detail='Inspect waveforms of trials for artifacts in either region and select the check box to the right each plot to include the trial in the curated set.\nLeave the check box unchecked otherwise to exclude noisy trials.\nClick "Complete curation" to write curated data to disk and exit program.\nClick "Exit" to exit the program without writing curated trials to disk.')

    def _get_data(self):
        PFC = np.load(self.PFCpath, allow_pickle=True).item()
        HPC = np.load(self.HPCpath, allow_pickle=True).item()

        self.PFCdata = PFC
        self.HPCdata = HPC

        assert PFC.keys() == HPC.keys()
        self.keys = list(PFC.keys())

        # for some reason when I included these functions as class members (or even in a main function)
        # plotting and writing pngs to disk causes tkinter to crash with the following message:

        ### *** Terminating app due to uncaught exception 'NSInvalidArgumentException', reason: '-[NSApplication _setup:]: unrecognized selector sent to instance 0x7fa541febc70'
        ### ...libc++abi: terminating due to uncaught exception of type NSException
        ### Abort trap: 6

        # and so I split up the functionality into a separate script and execute it in a new interpreter
        os.system(f'python3 _plot.py {self.PFCpath} {self.HPCpath}')

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta)), "units")

    def _checkall(self):
        for chk in self.checkboxes:
            chk.select()

    def _uncheckall(self):
        for chk in self.checkboxes:
            chk.deselect()

    def _cleartemp(self):
        if os.path.exists('temp/') and os.path.isdir('temp/'):
            shutil.rmtree('temp/')

    def _complete(self):
        curation = [var.get() for var in self.checkvars]
        filtered_keys = [k for k, check in zip(self.keys, curation) if check == 1]
        curated_PFC = dict(zip(filtered_keys, [self.PFCdata[k] for k in filtered_keys]))
        curated_HPC = dict(zip(filtered_keys, [self.HPCdata[k] for k in filtered_keys]))

        PFC_ofile = self.PFCpath.replace('.npy', '_curated.npy')
        HPC_ofile = self.HPCpath.replace('.npy', '_curated.npy')

        np.save(PFC_ofile, curated_PFC)
        np.save(HPC_ofile, curated_HPC)

        messagebox.showinfo(title=':-)', message='Curation complete!')

        self._cleartemp()
        sys.exit()

    def _exit(self):
        self._cleartemp()
        sys.exit()


def main():
    curationGUI(args.PFCpath, args.HPCpath)

if __name__ == '__main__':
    main()

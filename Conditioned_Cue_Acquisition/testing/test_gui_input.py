import os
import sys
import shutil
import tkinter as tk
import numpy as np
from tkinter import messagebox

class testGUI:
    def __init__(self):

        self.root = tk.Tk()
        self.root.title('LFP analysis with user input')

        self.frame = tk.Frame(self.root)
        self.frame.columnconfigure(0, weight=0)
        self.frame.columnconfigure(1, weight=1)

        self.l1 = tk.Label(self.frame, text='LFP analysis', font=('Helvetica', 16))
        self.l1.grid(row=0, column=0, columnspan=2)

        self.l2 = tk.Label(self.frame, text='path 1', font=('Helvetica', 12))
        self.l2.grid(row=1, column=0)
        self.input1 = tk.Text(self.frame, height=1, width=30, bg="white smoke", fg="black", highlightthickness=1, borderwidth=1)
        self.input1.grid(row=1, column=1, sticky=tk.E + tk.W)

        self.l3 = tk.Label(self.frame, text='path 2', font=('Helvetica', 12))
        self.l3.grid(row=2, column=0)
        self.input2 = tk.Text(self.frame, height=1, width=30, bg="white smoke", fg="black", highlightthickness=1, borderwidth=1)
        self.input2.grid(row=2, column=1, sticky=tk.E + tk.W)

        self.b1 = tk.Button(self.frame, text='Submit', command=self._show_inputs)
        self.b1.grid(row=3, column=0, columnspan=2)

        self.b2 = tk.Button(self.frame, text='Exit', command=self._on_closing)
        self.b2.grid(row=4, column=0, columnspan=2)

        ### select analysis type
        self.
        self.frame.pack(fill='x')

        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.mainloop()

    def _show_inputs(self):
        str1 = str(self.input1.get('1.0', tk.END))
        str2 = str(self.input2.get('1.0', tk.END))

        messagebox.showinfo(title='Message', message='These were the following inputs:', detail=f'\n{str1}\n{str2}')

    def _on_closing(self):
        self.root.destroy()

def main():
    testGUI()

if __name__ == '__main__':
    main()

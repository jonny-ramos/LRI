import os
import sys
import shutil
import argparse
import tkinter as tk
import numpy as np
import pandas as pd
from PIL import ImageTk, Image
from tkinter import messagebox

parser = argparse.ArgumentParser(description = 'simultaneously visually curate all trialized ephys data in a gui')
parser.add_argument('PFCpath', metavar='<path>', help='path to dict of PFC data to curate.')
parser.add_argument('HPCpath', metavar='<path>', help='path to dict of HPC data to curate.')
args = parser.parse_args()

class curationGUI:
    def __init__(self, PFCpath, HPCpath, rmsthresh=True):
        # init class members
        self.PFCpath = PFCpath
        self.HPCpath = HPCpath

        # this will get filled in later
        self.PFCdata = None
        self.HPCdata = None
        self.PFC_rmsfilt_data = None
        self.HPC_rmsfilt_data = None
        self.rats = None
        self.keys = None

        if rmsthresh:
            self._get_data(rmsthresh=True)
        else:
            self._get_data(rmsthresh=False)

        ### building the gui
        self.root = tk.Tk()

        #self.root.eval('tk::PlaceWindow . center')
        self.root.title('curation!')
        self.root.geometry('1450x2000')

        # menubar
        self.menubar = tk.Menu(self.root)

        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label='Check all', command=self._checkall)
        self.filemenu.add_command(label='Uncheck all', command=self._uncheckall)
        self.filemenu.add_command(label='Trial counts', command=self._trial_counts)
        self.filemenu.add_command(label='Trial n summary', command=self._trial_n_summary)
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

    def _rms(self, x):
        return np.sqrt(np.mean(np.array(x) ** 2))

    def _rat_rms(self, d_trials):
        # group data by rat (we only consider rms filtering within session)
        self.rats = np.unique([k.split('_trial')[0] for k in self.keys])
        keys = list(d_trials.keys())

        rat_rmss = []
        grouped_trials = []
        for rat in self.rats:
            rat_keys = [k for k in keys if rat in k]
            rat_trials = np.array([d_trials[k] for k in rat_keys])

            if not len(rat_trials) == 0:
                concat = np.concatenate(rat_trials)
                rat_rms = self._rms(concat)
                rat_rmss.append((rat, rat_rms))

                grouped_trials.append((rat, dict(zip(rat_keys, list(rat_trials)))))

        return dict(rat_rmss), dict(grouped_trials)

    def _windowed_rms(self, trial, step=500, n_samples=18000):
        return [self._rms(trial[i:i+step]) for i in np.arange(0, n_samples, step)]

    def _rms_filter(self, PFC_rms, HPC_rms, PFC_trials, HPC_trials, thresh=1.8):
        assert len(PFC_trials) == len(HPC_trials)
        assert PFC_trials.keys() == HPC_trials.keys()

        # PFC filter
        PFC_keys_filt = []
        for key in PFC_trials.keys():
            rmss = self._windowed_rms(PFC_trials[key])
            if np.all(np.array(rmss) / PFC_rms < thresh):
                PFC_keys_filt.append(key)

        # HPC filter
        HPC_keys_filt = []
        for key in HPC_trials.keys():
            rmss = self._windowed_rms(HPC_trials[key])
            if np.all(np.array(rmss) / HPC_rms < thresh):
                HPC_keys_filt.append(key)

        # sorted list of set intersection between PFC and HPC filters
        inter = set(PFC_keys_filt).intersection(set(HPC_keys_filt))
        inter = sorted(list(inter), key=lambda x: int(x.split('trial')[-1]))

        # building new dictionaries from filtered key list
        PFC_filt = dict([(k, PFC_trials[k]) for k in inter])
        HPC_filt = dict([(k, HPC_trials[k]) for k in inter])
        print(f'{len(PFC_trials) - len(PFC_filt)} trials removed via rms threshold (rms={thresh})')

        return PFC_filt, HPC_filt

    def _rmsfilt_data(self):
        d_PFC_rms, d_PFC = self._rat_rms(self.PFCdata)
        d_HPC_rms, d_HPC = self._rat_rms(self.HPCdata)

        # for each rat, apply rms threshold, build larger dict containing filtered data from all rats
        PFC_filt = {}
        HPC_filt = {}
        for rat in self.rats:
            PFC_rms, HPC_rms = d_PFC_rms[rat], d_HPC_rms[rat]
            PFC_trials, HPC_trials = d_PFC[rat], d_HPC[rat]

            rat_PFC_filt, rat_HPC_filt = self._rms_filter(PFC_rms, HPC_rms, PFC_trials, HPC_trials)
            PFC_filt = dict(**PFC_filt, **rat_PFC_filt)
            HPC_filt = dict(**HPC_filt, **rat_HPC_filt)

        self.PFC_rmsfilt_data = PFC_filt
        self.HPC_rmsfilt_data = HPC_filt

        # update self.keys after rms filtering
        assert self.PFC_rmsfilt_data.keys() == self.HPC_rmsfilt_data.keys()
        self.keys = list(self.PFC_rmsfilt_data.keys())

        # write rms filtered data to disk
        PFC_ofile = self.PFCpath.replace('.npy', '_rmsthresh.npy')
        HPC_ofile = self.HPCpath.replace('.npy', '_rmsthresh.npy')
        np.save(self.PFCpath.replace('.npy', '_rmsthresh.npy'), self.PFC_rmsfilt_data)
        np.save(self.HPCpath.replace('.npy', '_rmsthresh.npy'), self.HPC_rmsfilt_data)

        return PFC_ofile, HPC_ofile

    def _get_data(self, rmsthresh=True):
        self.PFCdata = np.load(self.PFCpath, allow_pickle=True).item()
        self.HPCdata = np.load(self.HPCpath, allow_pickle=True).item()

        assert self.PFCdata.keys() == self.HPCdata.keys()
        self.keys = list(self.PFCdata.keys())

        # for some reason when I included these functions as class members (or even in a main function)
        # plotting and writing pngs to disk causes tkinter to crash with the following message:

        ### *** Terminating app due to uncaught exception 'NSInvalidArgumentException', reason: '-[NSApplication _setup:]: unrecognized selector sent to instance 0x7fa541febc70'
        ### ...libc++abi: terminating due to uncaught exception of type NSException
        ### Abort trap: 6

        # and so I split up the functionality into a separate script and execute it in a new interpreter

        if rmsthresh:
            PFCfilt_path, HPCfilt_path = self._rmsfilt_data()
            os.system(f'python3 _plot.py {PFCfilt_path} {HPCfilt_path}')

            # once we've plotted, we don't need the thresholded set in memory anymore
            try:
                os.remove(PFCfilt_path)
                os.remove(HPCfilt_path)
            except OSError:
                pass

        else:
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

    def _trial_counts(self):
        n_total = len(self.PFCdata)
        curation = [var.get() for var in self.checkvars]
        n_cur = len([check for check in curation if check == 1])
        n_rmsthresh = len(self.PFC_rmsfilt_data)
        n_toss_thresh = n_total - n_rmsthresh
        n_toss_cur = n_rmsthresh - n_cur

        messagebox.showinfo(title=':-)', message='Curation complete!', detail=f'Total trial n: {n_total}\nrms thresholded trial n: {n_rmsthresh}\nCurated trial n: {n_cur}\n\n{n_toss_thresh} trials removed via rms thresholding\n{n_toss_cur} trials removed via curation\n total trials removed: {n_toss_thresh + n_toss_cur}')

    def _trial_n_summary(self):
        df = pd.DataFrame()
        df['rat_n'] = self.rats

        total_trial_ns = []
        for rat in self.rats:
            c = len([k for k in self.PFCdata.keys() if rat in k])
            total_trial_ns.append(c)
        df['total_trial_ns'] = total_trial_ns

        if not self.PFC_rmsfilt_data is None:
            thresh_trial_ns = []
            for rat in self.rats:
                c = len([k for k in self.PFC_rmsfilt_data.keys() if rat in k])
                thresh_trial_ns.append(c)
            df['rms_threshold_trial_ns'] = thresh_trial_ns

        curation = [var.get() for var in self.checkvars]
        filtered_keys = [k for k, check in zip(self.keys, curation) if check == 1]
        curation_trial_ns = []
        for rat in self.rats:
            c = len([k for k in filtered_keys if rat in k])
            curation_trial_ns.append(c)
        df['curated_trial_ns'] = curation_trial_ns

        df.to_csv('trial_n_summary.csv')


    def _complete(self):
        curation = [var.get() for var in self.checkvars]
        filtered_keys = [k for k, check in zip(self.keys, curation) if check == 1]
        curated_PFC = dict(zip(filtered_keys, [self.PFCdata[k] for k in filtered_keys]))
        curated_HPC = dict(zip(filtered_keys, [self.HPCdata[k] for k in filtered_keys]))

        PFC_ofile = self.PFCpath.replace('.npy', '_curated.npy')
        HPC_ofile = self.HPCpath.replace('.npy', '_curated.npy')

        np.save(PFC_ofile, curated_PFC)
        np.save(HPC_ofile, curated_HPC)

        self._trial_n_summary()
        self._trial_counts()

        self._cleartemp()
        sys.exit()



    def _exit(self):
        self._cleartemp()
        self.root.destroy()







def main():
    curationGUI(args.PFCpath, args.HPCpath)

if __name__ == '__main__':
    main()

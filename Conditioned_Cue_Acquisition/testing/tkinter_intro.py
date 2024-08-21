import tkinter as tk

root = tk.Tk()
root.geometry('500x500')
root.title('my gui')

# specify master first then named variables
label = tk.Label(root, text='Hello World!', font=('Arial', 18))
label.pack(padx=20, pady=20)

# again, we specify parent class first, then named variables
textbox = tk.Text(root, height=3, font=('Arial', 12)) # height = n_lines
textbox.pack(padx=10, pady=10)

# entry = tk.Entry(root)
# entry.pack()

# button
button = tk.Button(root, text='Click me!', font=('Arial', 36), activeforeground='white')
button.pack(padx=10, pady=10)

buttonframe = tk.Frame(root)
buttonframe.columnconfigure(0, weight=1)
buttonframe.columnconfigure(1, weight=1)
buttonframe.columnconfigure(2, weight=1)

btn1 = tk.Button(buttonframe, text='1', font=('Arial', 18), activeforeground='white')
btn1.grid(row=0, column=0, sticky=tk.W+tk.E)

btn2 = tk.Button(buttonframe, text='2', font=('Arial', 18), activeforeground='white')
btn2.grid(row=0, column=1, sticky=tk.W+tk.E)

btn3 = tk.Button(buttonframe, text='3', font=('Arial', 18), activeforeground='white')
btn3.grid(row=0, column=2, sticky=tk.W+tk.E)

btn4 = tk.Button(buttonframe, text='4', font=('Arial', 18), activeforeground='white')
btn4.grid(row=1, column=0, sticky=tk.W+tk.E)

btn5 = tk.Button(buttonframe, text='5', font=('Arial', 18), activeforeground='white')
btn5.grid(row=1, column=1, sticky=tk.W+tk.E)

btn6 = tk.Button(buttonframe, text='6', font=('Arial', 18), activeforeground='white')
btn6.grid(row=1, column=2, sticky=tk.W+tk.E)

btn7 = tk.Button(buttonframe, text='7', font=('Arial', 18), activeforeground='white')
btn7.grid(row=2, column=0, sticky=tk.W+tk.E)

btn8 = tk.Button(buttonframe, text='8', font=('Arial', 18), activeforeground='white')
btn8.grid(row=2, column=1, sticky=tk.W+tk.E)

btn9 = tk.Button(buttonframe, text='9', font=('Arial', 18), activeforeground='white')
btn9.grid(row=2, column=2, sticky=tk.W+tk.E)

anotherbtn = tk.Button(root, text='TEST', activeforeground='white')
anotherbtn.place(x=200, y=200, height=100, width=100)

# we have the pack layout in general, but within button frame we have grid layout
buttonframe.pack(fill='x')


root.mainloop()

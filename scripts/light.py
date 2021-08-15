from tkinter import Tk, Text

def change_color():
    current_color = box.cget("background")
    next_color = "black" if current_color == "white" else "white"
    box.config(background=next_color)
    root.after(59, change_color)

root = Tk()
box = Text(root, background="green")
box.pack()
change_color()
root.mainloop()
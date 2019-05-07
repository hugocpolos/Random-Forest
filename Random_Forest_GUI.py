try:
    from Tkinter import filedialog
    from Tkinter import *
except ImportError:
    from tkinter import filedialog
    from tkinter import *

from src.Dataset import Dataset
from src.DecisionTree import DecisionTree
import ntpath


class GUI_Class:
    """Classe que implementa a interface gráfica da
    aplicação"""

    def __init__(self, name, favicon):
        super(GUI_Class, self).__init__()
        self.root = Tk()
        self.app = Template(self.root)
        # self.root.resizable(False, False)
        self.root.title(name)
        try:
            self.root.call('wm', 'iconphoto',
                           self.root._w, PhotoImage(file=favicon))
        except (OSError, TclError, NameError):
            pass

    def start(self):
        self.root.mainloop()


class Template:
    def __init__(self, master=None):
        stdfont = ("Arial", "10")

        self.filename = StringVar(value='')
        self.delimiter_char = StringVar(value=";")

        # Choose dataset File Button
        okBtton = Button(master, text="Choose File", command=self.open_file,
                         width=12)
        okBtton["font"] = stdfont
        okBtton.grid(row=0, column=0, padx=5)

        # Display the filename - TODO
        self.filename_label = Label(master, text="No File Selected", width=32)
        self.filename_label["font"] = stdfont + ('bold',)
        self.filename_label.grid(row=0, column=1, sticky=E, padx=5, pady=20)

        # delimiter character
        delimiter_label = Label(master, text="delimiter Char: ")
        delimiter_label["font"] = stdfont
        delimiter_label.grid(row=1, column=0, sticky=W, padx=5, pady=20)

        delimiter_entry = Entry(
            master, width=1, textvariable=self.delimiter_char)
        delimiter_entry["font"] = stdfont
        delimiter_entry.grid(row=1, column=1, padx=5, pady=5)

        # Load Dataset Button
        okBtton = Button(master, text="Generate Tree", command=self.load_db,
                         width=12)
        okBtton["font"] = stdfont
        okBtton.grid(row=1, column=2, padx=5)

    def open_file(self):
        self.filename = filedialog.askopenfilename()
        self.filename_label['text'] = ntpath.basename(self.filename)

    def load_db(self):
        self.db = Dataset(
            self.filename, delimiter=self.delimiter_char.get(), ignore=[])
        self.tree = DecisionTree(self.db)
        print(self.tree)


if __name__ == '__main__':
    a = GUI_Class('Random Forest', '')
    a.start()

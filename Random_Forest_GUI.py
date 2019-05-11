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
        self.filename_meta = StringVar(value='')
        self.delimiter_char = StringVar(value=";")
        self.ignore_attrib = StringVar()
        self.T_value = IntVar()

        # Choose dataset File Button
        okBtton = Button(master, text="Dataset File", command=self.open_file,
                         width=12)
        okBtton["font"] = stdfont
        okBtton.grid(row=1, column=0, padx=5)

        # Display the filename
        self.filename_label = Label(master, text="No File Selected", width=32)
        self.filename_label["font"] = stdfont + ('bold',)
        self.filename_label.grid(row=1, column=1, sticky='W', padx=5, pady=5)

        # Choose dataset_meta File Button
        okBtton = Button(master, text="Metadata File",
                         command=self.open_file_meta, width=12)
        okBtton["font"] = stdfont
        okBtton.grid(row=2, column=0, padx=5)

        # Display the Metadata filename
        self.filename_meta_label = Label(
            master, text="No File Selected", width=32)
        self.filename_meta_label["font"] = stdfont + ('bold',)
        self.filename_meta_label.grid(
            row=2, column=1, sticky='W', padx=5, pady=5)

        # delimiter character
        delimiter_label = Label(master, text="delimiter Char: ")
        delimiter_label["font"] = stdfont
        delimiter_label.grid(row=3, column=0, sticky='W', padx=5, pady=20)

        delimiter_entry = Entry(
            master, width=1, textvariable=self.delimiter_char)
        delimiter_entry["font"] = stdfont
        delimiter_entry.grid(row=3, column=1, padx=5, pady=5, sticky="W")

        # Forest Size
        T_label = Label(master, text="Forest Length: ")
        T_label["font"] = stdfont
        T_label.grid(row=3, column=2, sticky='W', padx=5, pady=20)

        T_entry = Entry(
            master, width=5, textvariable=self.T_value)
        T_entry["font"] = stdfont
        T_entry.grid(row=3, column=3, padx=5, pady=5, sticky="W")

        # Error Message
        self.Error_Message = Label(
            master, text="", width=64)
        self.Error_Message["font"] = stdfont + ('bold',)
        self.Error_Message['fg'] = "#ff3c3c"
        self.Error_Message.grid(
            row=4, columnspan=4, padx=5, pady=0)

        ####

        # Load Dataset Button
        self.okBtton = Button(master, text="Generate Forest",
                              command=self.load_db_and_generate_forest,
                              width=12)
        self.okBtton["font"] = stdfont
        self.okBtton["state"] = DISABLED
        self.okBtton.grid(row=0, column=0, padx=5, pady=10)

        # Print Tree Button
        self.PrintButton = Button(master, text="Print Forest",
                                  command=self.print_forest,
                                  width=12)
        self.PrintButton["font"] = stdfont
        self.PrintButton["state"] = DISABLED
        self.PrintButton.grid(row=0, column=1, padx=5, pady=10)

        # Test Tree Button
        self.TestTree = Button(master, text="Test Forest",
                               command=self.print_forest,
                               width=12)
        self.TestTree["font"] = stdfont
        self.TestTree["state"] = DISABLED
        self.TestTree.grid(row=0, column=2,
                           columnspan=2, padx=5, pady=10)

    def open_file(self):
        self.filename = filedialog.askopenfilename()
        self.filename_label['text'] = ntpath.basename(self.filename)
        self.okBtton['state'] = 'normal'

    def open_file_meta(self):
        self.filename_meta = filedialog.askopenfilename()
        self.filename_meta_label['text'] = ntpath.basename(self.filename_meta)

    def load_db_and_generate_forest(self):
        self.forest = []
        # Load DB
        try:
            self.db = Dataset(
                self.filename,
                delimiter=self.delimiter_char.get(),
                metadata=self.filename_meta,
                bootstrap_n=self.T_value.get())
            self.Error_Message['text'] = ""
        except Exception as e:
            self.Error_Message['text'] = e
            return
        # Train Forest
        try:
            for training_set in self.db.training_set:
                self.forest.append(
                    DecisionTree(
                        training_set,
                        self.db.attributes,
                        self.db.predictclass,
                        self.db.numeric))
            self.Error_Message['text'] = ""
            self.PrintButton['state'] = 'normal'
        except Exception as e:
            self.Error_Message['text'] = e

    def print_forest(self):
        for tree in self.forest:
            print(tree)


if __name__ == '__main__':
    a = GUI_Class('Random Forest', '')
    a.start()

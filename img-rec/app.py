# Author: Derek Brown
import os
import pickle
from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askdirectory
import pandas as pd
from keras.preprocessing import image
import matplotlib.pyplot as plt
from csv import writer
import numpy as np
import train


# This is the main class which creates the UI and allows interactions
class App:
    def __init__(self, window):
        window.title('Alzheimer\'s Predictor')
        Label(window, text="Alzheimer\'s Predictor", font=('bold', 15)).pack()
        frame = Frame(window, width=360, height=225)
        frame.pack()

        Button(window, text='Retrain the Model', command=lambda: self.trainer()).place(x=50, y=50)
        Button(window, text='Accuracy of Model', command=lambda: self.acc_report()).place(x=198, y=150)
        Button(window, text='Load Data', command=lambda: self.load_data()).place(x=245, y=50)
        Button(window, text='Browse...', command=lambda: self.browse()).place(x=251, y=100)
        Button(window, text='Predict Results', command=lambda: self.prediction()).place(x=50, y=150)
        Button(window, text='Results Summary', command=lambda: self.create_pie_chart()).place(x=50, y=200)
        Button(window, text='Close', command=lambda: self.close()).place(x=269, y=200)

        self.textbox = Entry(window, width=30)
        self.textbox.place(x=50, y=104)
        self.textbox.insert(0, "Select a folder")
        self.status_txt = Label(window, foreground='red')
        self.status_txt.place(x=50, y=200)
        self.status_txt.config(text="Hello there!")
        self.status_txt.pack()

    # This will load the data and prepare it for training
    def load_data(self):

        prompt = messagebox.askokcancel("Confirmation",
                                        "You are about to reload the data, "
                                        "overwriting any previous data. This should only be done "
                                        "if a new dataset is being used to train the model.\n"
                                        "Select \'OK\' to continue")
        if not prompt:
            return
        self.status_txt.config(text="Loading data... Please wait.")
        try:
            train.data_loader()
            self.status_txt.config(text="Successfully loaded the data!")
        except:
            self.error_report()

    # This allows the model to be retrained
    def trainer(self):

        prompt = messagebox.askokcancel("Confirmation",
                                        "You are about to retrain the model, "
                                        "overwriting any previous models. "
                                        "This should only be done if the model needs to be retrained.\n"
                                        "Select \'OK\' to continue")
        if not prompt:
            return
        if not os.path.exists(r'resources/loaded_data.pkl'):
            self.status_txt.config(text="Please load the data before training the model")
            return
        try:
            self.status_txt.config(text="Training the model... Please wait.")
            train.train_model()
            self.status_txt.config(text="Successfully trained model!")
        except:
            self.error_report()

    # This allows the user to select a folder using Windows Explorer
    def browse(self):
        try:
            selection = askdirectory()
            self.textbox.delete(0, END)
            self.textbox.insert(0, selection)
        except:
            self.error_report()

    # This warns the user that the operation attempted was not successful
    def error_report(self):
        self.status_txt.config(text="Error! Something went wrong...")

    # This generates predictions for each image file in a folder
    def prediction(self):
        list_imgs = []
        try:
            folder = self.textbox.get()
            self.status_txt.config(text="Analyzing images... Please wait")
            loaded_model = pickle.load(open('resources/trained_model.pkl', 'rb'))
            if not os.path.exists(folder):
                self.status_txt.config(text="Folder could not be found. Please select a folder")
                return
            if os.path.exists(r'resources/report.csv'):
                prompt = messagebox.askyesno("Confirmation",
                                             "This action will overwrite the file \'report.csv\'.\n"
                                             "Do you wish to continue?")
                if not prompt:
                    return
            for file in os.listdir(folder):
                filename = os.path.join(folder, file)

                current_img = image.image_utils.load_img(filename, target_size=(1, 128, 128, 3))
                img_array = np.array(current_img, dtype='float32')
                img_array.resize(1, 128, 128, 3)

                prediction = loaded_model.predict(img_array)
                print(prediction)
                if prediction[0][0] < 0.5:
                    final_result = "Positive"
                else:
                    final_result = "Negative"
                try:
                    list_imgs.append((file, final_result))
                except:
                    list_imgs.append((file, 'Error!'))
            if len(list_imgs) > 0:
                self.create_table(list_imgs)

                with open(r'resources/report.csv', 'a+', newline='') as csv_file:

                    csv_write = writer(csv_file)

                    csv_report = open(r'resources/report.csv', 'w')
                    csv_report.truncate()
                    csv_report.close()
                    for row in list_imgs:
                        csv_write.writerow(row)

            else:
                messagebox.showinfo("No images were identified in folder")

            self.status_txt.config(text="Success! Results saved to report.csv")
        except:
            self.error_report()

    # This displays a table showing the values of each image predicted
    @staticmethod
    def create_table(t_list):
        fig, ax = plt.subplots()
        table = ax.table(cellText=t_list, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax.axis('off')
        plt.show()

    # this creates a line graph showing the accuracy of the model for training data
    def acc_report(self):
        accuracy_data = pickle.load(open('resources/model_accuracy.pkl', 'rb'))
        plt.plot(accuracy_data.history['accuracy'], 'bo--', label="Accuracy")
        plt.title("Accuracy")
        plt.ylabel("Accuracy (% in decimal format)")
        plt.xlabel("Epochs")
        plt.show()

    # this creates a pie chart displaying the totals for each diagnosis from report.csv
    def create_pie_chart(self):
        if not os.path.exists('resources/report.csv'):
            self.status_txt.config(text="No data - run a prediction to generate data")
            return
        with open('resources/report.csv', 'rt', encoding="utf8") as csv_obj:
            r_list = list(csv_obj)
            rows = []
            for item in r_list:
                rows.append(item.split(','))
            if len(rows) < 1:
                self.status_txt.config(text="No data - run a prediction to generate data")
                return
            chart = pd.DataFrame(rows, columns=['Name', 'Prediction'])

            counter = chart['Prediction'].value_counts()
            plt.pie(counter, labels=counter.index, autopct='%.2f')
            plt.show()
            self.status_txt.config(text="Please select an option")

    # this closes the program
    def close(self):
        prompt = messagebox.askyesno("Close Program?",
                                     "Do you want to close the program?")
        if not prompt:
            return
        exit()


# these lines of code create the window for the UI
fenster = Tk()
fenster.eval('tk::PlaceWindow . center')
fenster.resizable()
app = App(fenster)

fenster.mainloop()

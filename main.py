"""
Arthur Rohaert
Tue Jul 19 15:58:42 2022
"""

# %%------------------------------------------------------- Import the packages
###############################################################################

import os
import threading
from kivy.app import App
from kivy.factory import Factory
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import StringProperty, ObjectProperty, BooleanProperty #, AliasProperty
from kivy.uix.popup import Popup
from kivy.config import Config
from execution import try_execution

# %%------------------------------------------------------ Build main interface
###############################################################################

# initial window size
Config.set("graphics", "width", 500)
Config.set("graphics", "height", 1000)
Config.set("graphics", "left", 1)

class Root(BoxLayout):
    
    #################################################### GUI related variables
    directory = StringProperty(os.environ["HOMEPATH"])
    get_files_locally = BooleanProperty(True)
    path = StringProperty(os.environ["HOMEPATH"])
    disable_individual_scatters = BooleanProperty(False)
    disable_individual_series = BooleanProperty(False)
    disable_scatter = BooleanProperty(False)
    disable_fits = BooleanProperty(False)
    disable_all_figures = BooleanProperty(False)
    disable_execution = BooleanProperty(True)
    GS = BooleanProperty(True)
    UW = BooleanProperty(True)
    DR = BooleanProperty(True)
    DA = BooleanProperty(True)
    VA = BooleanProperty(True)
    DC = BooleanProperty(True)
    UD = BooleanProperty(False)
    workload = 0
    progress = 0
    percentage = 0
    progress_message = StringProperty("Progress: 0%")
    executing = BooleanProperty(False)

    ################################################## Input and output options
    # pop-up to choose the directory
    def OpenDirectoryChooser(self):
        content = DirectoryPopup(select_directory = self.select_directory)
        self._popup = Popup(title="Open a directory (double click) and choose 'Select'",
                            content=content, size_hint=(0.9,0.7))
        self._popup.open()

    # selection of the directory
    def select_directory(self, path, filename):
        self.directory = path
        self._popup.dismiss()

    # pop-up to choose the input file
    def OpenFileChooser(self):
        content = FilePopup(select_file = self.select_file)
        self._popup = Popup(title="Select a file",
                            content=content, size_hint=(0.9,0.7))
        self._popup.open()

    # selection of the input file
    def select_file(self, path, filename):
        self.path = filename[0]
        self._popup.dismiss()

    # input option: local file or online database
    def data_option(self, local_files):
        self.get_files_locally = local_files

    ############################################################## Plot options
    def plot_individual_scatters(self, state):
        self.disable_individual_scatters = (True if state == "normal" else False)
        self.plot_any_figure()

    def plot_individual_series(self, state):
        self.disable_individual_series = (True if state == "normal" else False)
        self.plot_any_figure()

    def plot_scatter(self, state):
        self.disable_scatter = (True if state == "normal" else False)
        self.plot_any_figure()

    def plot_fits(self, state):
        self.disable_fits = (True if state == "normal" else False)
        self.plot_any_figure()

    def plot_any_figure(self):
        self.disable_all_figures = all([self.disable_individual_scatters,
                                        self.disable_individual_series,
                                        self.disable_scatter,
                                        self.disable_fits])
    
    def manual_models(self):
        if self.ids.UD.active:
            self.UD = True
        else:
            self.UD = False

    #################################################### Gathering all settings
    def execute(self):
        # gathering general project information
        input_information = {}
        input_information["directory"] = self.directory
        
        # gathering information related to loading a local datasheet
        input_information["download_data"] = not self.get_files_locally
        input_information["file"] = self.path
        
        # gathering information related to downloading the data
        input_information["username"] = self.ids.username.text
        input_information["password"] = self.ids.password.text
        detectors_list = (self.ids.detectors.text.replace(" ", "")).split("\n")
        detectors_dict = {el.split(":")[0]:int(el.split(":")[1]) for el in detectors_list}        
        input_information["detectors"] = detectors_dict
        dates_list = (self.ids.dates.text.replace(" ", "")).split("\n")
        dates_dict = {}
        i = 0
        while i < len(dates_list):
            if dates_list[i] != "":
                if dates_list[i][-1] == ":":
                    key = dates_list[i][:-1]
                    value = []
                    while i+1 < len(dates_list):
                        i += 1
                        if dates_list[i] != "":
                            if dates_list[i][-1] != ":":
                                value.append(dates_list[i])
                            else:
                                break
                    dates_dict[key] = value
                else:
                    i += 1
            else:
                i += 1
        input_information["dates"] = dates_dict
        
        # gathering information related to plot options
        generate_plots = [not self.disable_individual_scatters,
                          not self.disable_individual_series,
                          not self.disable_scatter,
                          not self.disable_fits]
        input_information["generate_plots"] = generate_plots
        input_information["user-defined models"] = self.UD
        if not self.UD:
            fit_models = [self.ids.GS.active, self.ids.UN.active,
                          self.ids.DR.active, self.ids.DA.active,
                          self.ids.VA.active, self.ids.DE.active,
                          self.ids.HC.active, self.ids.CH.active]
            input_information["fit_models"] = fit_models
        plot_range = [float(self.ids.density.text),
                      float(self.ids.speed.text),
                      float(self.ids.flow.text)]
        input_information["plot_range"] = plot_range
        formats = []
        if self.ids.pdf.active:
            formats.append('pdf')
        if self.ids.png.active:
            formats.append('png')
        if self.ids.svg.active:
            formats.append('svg')
        if len(formats) == 0:
            self.ids.pdf.active = True
            formats = ['pdf']
        input_information["format"] = formats
        
        # estimating workload
        self.workload = 0
        self.workload += (30 if self.get_files_locally else 270)
        if not self.disable_individual_scatters: self.workload += 200
        if not self.disable_individual_series: self.workload += 200
        if not self.disable_scatter: self.workload += 30
        if not self.disable_fits:
            if self.UD:
                self.workload += 200
            else:
                self.workload += sum(fit_models)*50
        
        # execute!
        self.progress = 0
        self.percentage = 0
        self.ids.probar.value = 0
        self.executing = True
        self.ids.execute_button.markup = True
        self.ids.execute_button.text = "[b][color=9CE4FF] P L E A S E    W A I T  . . .[/color][/b]"
        x = threading.Thread(target=try_execution, args=(self,input_information))
        x.start()

    ################################################### Progress update for gui
    def update(self, value, status_update=None):
        self.progress = self.progress + value
        self.progress_message = "Progress: "+ str(int(self.percentage)) + "%"
        self.percentage = min(int(100*self.progress/self.workload),100)
        self.ids.probar.value = self.percentage
        if status_update != None:
            self.ids.status.text = "Status: " + status_update

# %%------------------------------------------------------------- Build pop-ups
###############################################################################
        
class DirectoryPopup(FloatLayout):
    initial_path = StringProperty(os.environ["HOMEPATH"])
    select_directory = ObjectProperty(None)

class FilePopup(FloatLayout):
    initial_path = StringProperty(os.environ["HOMEPATH"])
    select_file = ObjectProperty(None)

class TDAApp(App):
    pass

# %%------------------------------------------------------------------ Run code
###############################################################################

TDAApp().run()
Factory.register('Root', cls=Root)
Factory.register('DirectoryPopup', cls=DirectoryPopup)

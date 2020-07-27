import numpy as np
from calciumflexanalysis import calcium_flex as cal
from collections import defaultdict
import pandas as pd 
import matplotlib.pyplot as plt

class CaFlexGroup:
    """Class used for the analysis of multiple Calcium Flex well plates.
    
    :param caflexplates: List of caflexplates to combine, generated from CaFlexPlate class
    :type caflexplates: list of calciumflexanalysis.calcium_flex.CaFlexPlates

    """
    
    def __init__(self, caflexplates = []):
    
        self.caflexplates = caflexplates
        self.grouplist = ['Protein', 'Type', 'Compound', 'Concentration', 'Concentration Units']

        self.titles = {}
        self.plate_maps = {}
        self.data = {'ratio':{}}
        
        inject_list = []
        # iterate through each plate and update attributes using predefined caflexanalysis methods
        for key, val in enumerate(self.caflexplates):
            # titles (nice ref for the user)
            self.titles["plate_{}".format(key+1)] = val.title
            # update processed data w/ ratios
            self.data['ratio']["plate_{}".format(key+1)] = val.processed_data['ratio']
            # dictionary of plate maps for each plate
            self.plate_maps["plate_{}".format(key+1)] = val.plate_map
            
            # append list with injection times for each plate
            inject_list.append(val.inject)
        
        # mean inject across all plates (this might be changed)
        self.inject = np.array(inject_list).mean()
        
    def visualise_plates(self, share_y, export = False, title = "", colormap = 'Dark2_r',
             colorby = 'Type', labelby = 'Type', dpi = 200):
        """Returns color-coded and labelled plots of the data collected for each well of each well plate.

        :param share_y: 'True' sets y axis the same for all plots
        :type share_y: bool
        :param export: If 'True' a .png file of the figure is saved, default = False
        :type export: bool
        :param title: Sets the title of the figure, optional
        :type title: str
        :param colormap: Sets the colormap for the color-coding, default = 'Dark2_r'
        :type colormap: str
        :param colorby: Chooses the parameter to color code by, for example 'Type', 'Contents', 'Concentration', 'Compound', 'Protein', 'Concentration Units', default = 'Type'
        :type colorby: str
        :param labelby: Chooses the parameter to label code by, for example 'Type', 'Contents', 'Concentration', 'Compound', 'Protein', 'Concentration Units', default = 'Type'
        :type labelby: str
        :param dpi: Size of the figure, default = 200
        :type dpi: int
        :return: Figure of plotted data for each well of the well plate described in plate_map_file
        :rtype: figure
        """
        plates = self.caflexplates
        for key, val in enumerate(plates):
            if title == "":
                title = "Plate {}\n{}".format(key+1, val.title)
            val.visualise_assay(share_y, export, title, colormap, colorby, labelby, dpi)
            
    def see_plates(self, title = "", export = False, colormap = 'Paired', colorby = 'Type', labelby = 'Type', dpi = 100):
        """Returns a visual representation of each plate map.
    
        The label and colour for each well can be customised to be a variable, for example 'Compound', 'Protein', 'Concentration', 'Concentration Units', 'Contents' or 'Type'. The size of the plate map used to generate the figure can be either 6, 12, 24, 48, 96 or 384. 
        :param size: Size of platemap, 6, 12, 24, 48, 96 or 384, default = 96
        :type size: int    
        :param export: If 'True' a .png file of the figure is saved, default = False
        :type export: bool
        :param title: Sets the title of the figure, optional
        :type title: str
        :param colormap: Sets the colormap for the color-coding, default = 'Paired'
        :type colormap: str
        :param colorby: Chooses the parameter to color code by, for example 'Type', 'Contents', 'Concentration', 'Compound', 'Protein', 'Concentration Units', default = 'Type'
        :type colorby: str
        :param labelby: Chooses the parameter to label code by, for example 'Type', 'Contents', 'Concentration', 'Compound', 'Protein', 'Concentration Units', default = 'Type'
        :type labelby: str
        :param dpi: Size of the figure, default = 150
        :type dpi: int
        :return: Visual representation of the plate map.
        :rtype: figure
        """
        plates = self.caflexplates
        for key, val in enumerate(plates):
            if title == "":
                title = "Plate {}\n{}".format(key+1, val.title)
            val.see_plate(title, export, colormap, colorby, labelby, dpi)
            
    def baseline_correct(self):
        """Baseline corrects 'ratio' data for each well using the pre-injection time points."""
        self.data['baseline_corrected'] = {}
        for key, val in enumerate(self.caflexplates):
            val.baseline_correct()
            self.data['baseline_corrected']["plate_{}".format(key+1)] = val.processed_data['baseline_corrected']
        print("baseline corrected!")
        
    def get_window(self, data_type):
        """Finds the lowest overall mean gradient for across the ten time point window post injection for the plates
        
        :param data_type: Data series to calculate plateau, either 'ratio' or 'baseline_corrected'
        :type data_type: str
        :return: Tuple containing start and end index of plateau window
        :rtype: (int, int)
        """
        plates = self.caflexplates
        gradients = {}
        
        for key, val in enumerate(plates):
            g = val.get_gradients(data_type)
            
            # keys for each plate are numbered by key this time - easier for unpacking
            gradients[key] = g
            
        # collect gradients for each window in each plate into single dictionary using default dict
        windows = defaultdict(list)
        for key, val in gradients.items():
            for k, v in val.items(): # unpack dictionary of dictionaries
                windows[k].append(v) # where multiple plates have the same window, the resulting dict value will be a list of those gradients
        
        # take means of each window
        mean_windows = {}
        for key, val in windows.items():
            mean_windows[key] = np.array(val).mean()
                
        # get minimum gradient index window across all plates and update self.window
        self.window = (min(mean_windows, key = mean_windows.get))
        
        # update windows for each plate
        for key, val in enumerate(plates):
            val.window = self.window
        
        return self.window
    
    def def_window(self, time, data_type):
        """Manually sets each plateau window.
        
        :param time: Time point at start of window
        :type time: int
        :param data_type: Data to set window on, either 'ratio' or 'baseline_corrected'
        :type data_type: str
        :return: Tuple containing start and end index of plateau window
        :rtype: (int, int)
        """
        plates = self.caflexplates
        temp = []
        for key, val in enumerate(plates):
            val.def_window(time, data_type)
            temp.append(val.window)
        if all(x == temp[0] for x in temp) == True:
            self.window = temp[0]
            print("all windows equal, self.window updated")
            return self.window
        else:
            raise ValueError("Time points are not equal")
        
    def group_data(self, data_type):
        """Groups data from each plate of desired type (either ratio or baseline_corrected) into single dataframe.
        
        :param data_type: Data to be groupe, either 'ratio' or 'baseline_corrected'
        :type data_type: str
        :return: Dictionary of dataframes
        :rtype: {str:pandas.DataFrame, str:pandas.DataFrame}
        """
        plates = self.caflexplates
        group_list = self.caflexplates
        
        data_list = [] # collect all data in list, then concatenate dfs, take means for each condition
        time_list = [] # same for time (sem not required)
        # get data for each plate in plates_list
        for key, val in enumerate(plates):
            plate_map = val.plate_map
            # extract data, combine with the plate's plate map, append data_list
            mapped = plate_map.fillna('none').join(val.processed_data[data_type]['data'])
            data_list.append(mapped)

        # repeat for time:
        for key, val in enumerate(plates):
            plate_map = val.plate_map
            # extract data, combine with the plate's plate map, append data_list
            mapped = plate_map.fillna('none').join(val.processed_data[data_type]['time'])
            time_list.append(mapped)

        # concat data and time - all data for every well now in one df for data and time
        all_data = pd.concat(data_list, ignore_index = True)
        all_time = pd.concat(time_list, ignore_index = True)
        
        self.data[data_type]['grouped'] = {'data':all_data, 'time': all_time}
        
        print("self.data updated. See self.data[{}]['grouped']".format(data_type))
        
    def plot_conditions(self, data_type, plate_number = True, activator = " ", show_window = False, dpi = 120, title = "", error = False, control = True, cmap = "winter_r", window_color = 'hotpink', proteins = [], compounds = []):
        """Plots each mean condition versus time, for either each plate or over all plates, for each compound and protein.
        
        If no title is desired, set title to " ".
        

        :param plate_number: If True, plate number is added to each plot title, default = True
        :type plate_number: bool
        :param data_type: Data to be plotted, either 'ratio' or 'baseline_corrected'
        :type data_type: str
        :param show_window: If 'True', shows the window from which the plateau for each condition is calculated, default = False 
        :type show_window: bool
        :param dpi: Size of figure, default = 120
        :type dpi: int
        :param title: Title of plot ADD LIST OF TITLES?
        :type title: str
        :param error: If True, plots error bars for each mean condition, default = False
        :type error: bool
        :param control: If True, plots control data, default = True
        :type control: bool
        :param cmap: Colormap to use as the source of plot colors
        :type cmap: str
        :param window_color: Color of the plateau window, default = 'hotpink'
        :type window_color: str
        :return: Figure displaying each mean condition versus time
        :rtype: fig
        """
        grouplist = self.grouplist
        
        if combine == False: 
            for key, val in enumerate(self.caflexplates):
                # sort titles
                if plate_number == True: # show 
                    if title == "":
                        Title = "Plate {}\n{}".format(key+1, val.title)
                    else:
                        Title = "Plate {}\n{}".format(key+1, title)
                else:
                    if title == "":
                        Title = val.title
                val.plot_conditions(data_type, activator, show_window, dpi, Title, error, control, cmap, window_color, proteins, compounds)
                
    def amplitude(self, data_type):
        """Calculates response amplitude for each condition, for either each plate or across all plates.
        
        :param data_type: Data to use to calculate amplitudes, either 'ratio' or 'baseline_corrected'
        :type data_type: str
        :param combine: Generate amplitudes for each plate or across all plates, default = False
        :type combine: bool
        """

        for key, val in enumerate(self.caflexplates):
            val.amplitude(data_type)
            print("self.processed_data['plateau']['data'] updated for plate {}.".format(key+1))
                
    def mean_amplitude(self, use_normalised = False, combine = False):
        """Returns mean amplitudes and error for each condition, for either each plate or across all plates.
        
        The user must run the normalise method before attempting to get the mean amplitudes of the normalised amplitudes.
        
        :param use_normalised: If True, uses normalised amplitudes, default = False
        :type use_normalised: bool
        :return: Mean amplitudes and error for each condition
        :rtype: Pandas DataFrame
        :param combine: Generate mean amplitudes for each plate or across all plates, default = False
        :type combine: bool
        """
        # check data_type == 'ratio' or 'baseline_corrected'
        self.data['mean_amplitudes'] = {}
        lst = []
        
        if combine == False:
            for key, val in enumerate(self.caflexplates):
                mean_amps = val.mean_amplitude(use_normalised) # get mean amps for each plate
                self.data['mean_amplitudes'][key] = mean_amps # update self.data
                print("self.data['mean_ampltitudes'][{}] updated".format(key))
                
        if combine == True:
            for key, val in enumerate(self.caflexplates):
                # combine ampltitude data for each plate with its corresponding plate map
                mapped = val.plate_map.fillna(-1).join(val.processed_data['plateau']['data'])
                if use_normalised == True :
                    # get normalised data if specified
                    mapped = val.plate_map.fillna(-1).join(val.processed_data['plateau']['data_normed'])
                # collect resulting dataframes    
                lst.append(mapped)
            # concatenate resulting dataframes        
            mapped = pd.concat(lst)
            # get only valid wells
            group = mapped[mapped.Valid == True]
            # drop columns which cause errors w/ grouby operations
            group.drop(['Valid', 'Column'], axis = 1, inplace = True)
            # get mean amps for each condition across all plates
            mean_response = group.groupby(self.grouplist).mean().reset_index()
            # get errors
            if use_normalised == False:
                mean_response['Amplitude Error'] = group.groupby(self.grouplist).sem().reset_index().loc[:, 'Amplitude']
            else:
                mean_response['amps_normed_error'] = group.groupby(self.grouplist).sem().reset_index().loc[:, 'amps_normed']
            # drop empty rows
            mean_response.drop(mean_response[mean_response['Type'] == 'empty'].index)
            # update self.data
            self.data['mean_amplitudes'] = mean_response

            return self.data['mean_amplitudes']
    
    
    def collect_curve_data(self, plot_func, use_normalised, n, proteins, compounds, **kwargs):
        """Updates self.plot_data with the data from all the plates."""
        
        mean_amps = self.mean_amplitude(combine = True)
        # use static method in calcium_flex to get curve_data
        curve_data = cal.CaFlexPlate._gen_curve_data(mean_amps, plot_func, use_normalised, n, proteins, compounds, **kwargs)
        
        self.plot_data = curve_data
        
        return self.plot_data
    
    
    def plot_curve(self, plot_func, combine_plates = False, combine = False, plate_number = True, activator = " ", use_normalised = False, type_to_plot = 'compound', title = ' ', dpi = 120, n = 5, proteins = [], compounds = [], error_bar = True, cmap = "Dark2", show_top_bot = False, **kwargs):
        """Plots fitted curve, for either each plate or a combined plot using logistic regression with errors and IC50/EC50 values.
        
        :param plot_func: Plot function to use, either ic50 or ec50
        :type plot_func: str
        :param combine_plates: Combines all plots across all plates onto the same graph, default = False
        :type combine_plates = bool
        :param combine: Combines different proteins and compounds on same plate to the same plot, default = False
        :type combine: bool
        :param activator: Activator injected into assay, default = ""
        :type activator: str
        :param use_normalised: If True, uses normalised amplitudes, default = False
        :type use_normalised: bool
        :param type_to_plot: Type of condition to plot, default = 'compound'
        :type type_to_plot: str
        :param title: Choose between automatic title or insert string to use, default = 'auto'
        :type title: str
        :param dpi: Size of figure
        :type dpi: int
        :param n: Number of concentrations required for plot
        :type n: int
        :param proteins: Proteins to plot, defaults to all
        :type proteins: list
        :param compounds: Compounds to plot, defaults to all
        :type compounds: list
        :param show_top_bot: 'True' shows the top and bottom curve fitting values in the legend
        :type show_top_bot: bool
        :param **kwargs: Additional curve fitting arguments
        :return: Figure with fitted dose-response curve
        :rtype: fig
        """
        # plot each plate separately (combine can be on or off)
        if combine_plates == False:
            for key, val in enumerate(self.caflexplates):
                val.plot_curve(plot_func, use_normalised, n, proteins, compounds, error_bar, cmap, combine, activator, title, dpi, show_top_bot, **kwargs) # update with type_to_plot
            
        # combine data from all plates (combine can still separate proteins/compounds)
        if combine_plates == True:
            curve_data = self.collect_curve_data(plot_func, use_normalised, n, proteins, compounds, **kwargs)
            
            # use static method in calcium_flex to plot
            cal.CaFlexPlate._plot_curve(curve_data, plot_func, use_normalised, n, proteins, compounds, error_bar, cmap, combine, activator, title, dpi, show_top_bot)
    
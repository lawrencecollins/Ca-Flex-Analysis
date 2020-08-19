import numpy as np
from calciumflexanalysis import calcium_flex as cal
from collections import defaultdict
import pandas as pd 
import matplotlib.pyplot as plt
from datetime import date
import copy

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
                Title = "Plate {}\n{}".format(key+1, val.title)
            else:
                Title = "Plate {}\n{}".format(key+1, title)
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
                Title = "Plate {}\n{}".format(key+1, val.title)
            else:
                Title = "Plate {}\n{}".format(key+1, title)
            try:
                val.see_plate(Title, export, colormap, colorby, labelby, dpi)
            except:
                print("Check plate {}".format(key+1))
                
    def baseline_correct(self):
        """Baseline corrects 'ratio' data for each well using the pre-injection time points."""
        self.data['baseline_corrected'] = {}
        for key, val in enumerate(self.caflexplates):
            try:
                val.baseline_correct()
                print("Plate {}".format(key+1))
                self.data['baseline_corrected']["plate_{}".format(key+1)] = val.processed_data['baseline_corrected']
                
            except:
                print("Baseline correction for plate {} failed".format(key+1))
                
        
        
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
        
    def plot_conditions(self, data_type, plate_number = True, activator = " ", show_window = False, dpi = 120, title = "", error = False, control = ['control'], cmap = "winter_r", window_color = 'hotpink', proteins = [], compounds = [], marker = 'o', unique_markers = False, marker_list = ["o", "^", "s", "D", "p", "*", "v"], show_control = True):
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
        :param marker: Marker type, default = '-o'
        :type marker: str
        :param unique_markers: If True, plots each condition as a black line with a unique marker, default = False
        :type unique_markers: bool
        :param marker_list: List of marker symbols to use when unique_markers = True, default = ["o", "^", "s", "D", "p", "*", "v"]
        :type marker_list: [str]
        :return: Figure displaying each mean condition versus time
        :rtype: fig
        """
        grouplist = self.grouplist

        for key, val in enumerate(self.caflexplates):
            try:
                # sort titles
                if plate_number == True: # show 
                    if title == "":
                        Title = "Plate {}\n{}".format(key+1, val.title)
                    else:
                        Title = "Plate {}\n{}".format(key+1, title)
                else:
                    if title == "":
                        Title = val.title
                val.plot_conditions(data_type, activator, show_window, dpi, Title, error, control, cmap, window_color, proteins, compounds, marker, unique_markers, marker_list, show_control)
            except:
                print("Plate {} plot failed".format(key+1))
                
    def amplitude(self, data_type):
        """Calculates response amplitude for each condition, for each plate.
        
        Don't worry about specifying whether you want to collate data yet - it'll do both.
        
        :param data_type: Data to use to calculate amplitudes, either 'ratio' or 'baseline_corrected'
        :type data_type: str
        
        """
        for key, val in enumerate(self.caflexplates):
            try:
                val.amplitude(data_type)
                print("self.processed_data['plateau']['data'] updated for plate {}.".format(key+1))
            except:
                print("Plate {} amplitude calculation failed.".format(key+1))
            
        # collate data processed data from all plates and update self.data
        self.group_data(data_type)
        amp = self.data[data_type]['grouped']['data'].iloc[:, self.window[0]:self.window[1]].to_numpy()
        # easiest means to do this is to just calculate the means again - keeps everything in the same order. 
        amp_mean = np.mean(amp, axis = 1)
        plateaus = pd.DataFrame(amp_mean, columns = ['Amplitude'], index = self.data[data_type]['grouped']['data'].index)
        
        # get plate map info
        cols = self.data[data_type]['grouped']['data'].columns
        cols_fltr = [i for i in cols if type(i) == str] # extract plate infor excluding data
        grouped_map = self.data[data_type]['grouped']['data'][cols_fltr]
        self.data['plateau'] = {} 
        
        # merge plate map details with amps
        self.data['plateau']['data'] = grouped_map.join(plateaus) 
        
        return self.data['plateau']['data']
        
        print("self.data['plateau']['data'] updated")
        
                
    def mean_amplitude(self, use_normalised = False, combine = True):
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
        
        if combine == False:
            for key, val in enumerate(self.caflexplates):
                mean_amps = val.mean_amplitude(use_normalised) # get mean amps for each plate
                self.data['mean_amplitudes'][key] = mean_amps # update self.data
                print("self.data['mean_ampltitudes'][{}] updated".format(key))
                
        if combine == True:
            if use_normalised == False:
                group = self.data['plateau']['data']
            else:
                group = self.data['plateau']['data_normed'] # UPDATE NORMALISE FOR COMBINE
            # get valid data
            group = group[group.Valid == True]
            # drop cols which cause errors w/ groupby operations
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
    
    def normalise(self, combine = True):
        """Normalises amplitudes to mean control amplitude. 
        
        If combine = True, normalises to the mean control amplitude across all the plates. Updates either each caflexplate object or the caflexgroup object.
        
        :param combine: If true, normalises the response amplitudes to the mean control across all the plates
        :type combine: bool
        """
        if combine == False:
            for key, val in enumerate(self.caflexplates):
                val.normalise()
                print("Plate {} normalised".format(key+1))
                
        if combine == True:
            # get mean control amplitude
            mean_amps = self.mean_amplitude(use_normalised = False, combine = True)
            amps = self.data['plateau']['data']['Amplitude']
            control_amp = mean_amps[mean_amps['Type'] == 'control']['Amplitude'].mean()
            # get plate map infor for every well across every plate
            platemap = self.data['plateau']['data'].loc[:, 'Well ID': 'Valid']
            # normalise to mean control
            normed = ((amps*100) / control_amp).to_frame().rename(columns = {'Amplitude':'amps_normed'})
            # update self.data
            self.data['plateau']['data_normed'] = platemap.join(normed)
            print("Collated data normalised to mean control. See self.data['plateau']['data_normed']")
            return self.data['plateau']['data_normed']
    
    def collate_curve_data(self, plot_func, use_normalised, n, proteins, compounds, **kwargs):
        """Updates self.plot_data with the data from all the plates."""
        
        mean_amps = self.mean_amplitude(combine = True, use_normalised = use_normalised)
        # use static method in calcium_flex to get curve_data
        curve_data = cal.CaFlexPlate._gen_curve_data(mean_amps, plot_func, use_normalised, n, proteins, compounds, **kwargs)
        
        self.plot_data = curve_data
        
        return self.plot_data
    
    
    def plot_curve(self, plot_func, combine_plates = False, combine = True, plate_number = True, activator = " ", use_normalised = False, type_to_plot = 'compound', title = ' ', dpi = 120, n = 5, proteins = [], compounds = [], error_bar = True, cmap = "Dark2", show_top_bot = False, conc_units = 'M', **kwargs):
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
        :param conc_units: Units displayed on x axis of graph
        :type conc_units: str
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
            curve_data = self.collate_curve_data(plot_func, use_normalised, n, proteins, compounds, **kwargs)
            
            # add curve data as object attribute
            self.curve_data = curve_data
            for k in self.curve_data.keys():
                self.curve_data[k]['plot_func'] = plot_func
            
            
            # use static method in calcium_flex to plot
            cal.CaFlexPlate._plot_curve(curve_data, plot_func, use_normalised, n, proteins, compounds, error_bar, cmap, combine, activator, title, dpi, show_top_bot, conc_units)
    
    def export_data(self, combine_plates = False, title = ""):
        """Exports raw and processed data to an excel file. 
        
        The excel file will be named after the 'self.title' parameter unless otherwise specified. Combining the data will calulate and generate the curve fit values that correspond to those used when combine_plates = True in plot_conditions. It is recommended that the user set the title if combine = True.
        
        :param combine_plates: Combines data from each plate to the same document, default = False
        :type combine_plates: bool
        :param title: Title of the excel file, default = self.title + '_data_export.xlsx'
        :type title: str
        :return: Excel document
        :rtype: .xlsx
        """
        if combine_plates == False:
            for key, val in enumerate(self.caflexplates):
#                 if title[-5:] != '.xlsx':
#                     title = title + '.xlsx'
                if title == "":
                    val.export_data()
                else:
                    val.export_data("{}_{}".format(key, title))
                print("Plate {} data exported as {}".format(key+1, title))
        if combine_plates == True:
            # set title
            if title == "":
                title = date.today().strftime("%d-%m-%Y") + "_combined_data_" + '.xlsx'
                
            # get curve data 
            curve_dict = {}

            if hasattr(self, 'curve_data'):
                for key, val in self.curve_data.items(): # get curve data for each protein/compound combo
                    curve = copy.deepcopy(val) # get deep copy - prevents original dict from being edited
                    curve['popt_keys'] = ['top', 'bottom', curve['plot_func'], 'hill'] # add popt index
                    for i in ['compound', 'protein', 'c50units', 'plot_func']: # delete irrelevent keys
                        curve.pop(i)
                    curve_df = pd.DataFrame.from_dict(curve, orient = 'index')

                    curve_dict[key] = curve_df
            
                    
            # unpack data dictionary and upload to excel:
            with pd.ExcelWriter(title) as writer:
                for k1, v1 in self.data.items():
                    if isinstance(v1, pd.DataFrame):
                        v1.to_excel(writer, sheet_name = k1)
                    for k2, v2 in v1.items():
                        if isinstance(v2, pd.DataFrame):
                            v2.to_excel(writer, sheet_name = k1 + "_" + k2)
                        for k3, v3 in v2.items():
                            if isinstance(v3, pd.DataFrame):
                                v3.to_excel(writer, sheet_name = k1 + "_" + k2 + "_" + k3)
                # export curve data for each protein/compund combination
                for key, val in curve_dict.items():
                    key = key.translate({ord(i): None for i in "[]:*?/" }) # get rid of invalid characters
                    val.to_excel(writer, sheet_name = key + '_curve_data')
                    
                    
            print("Combined plate data exported as {}".format(title))
            
            
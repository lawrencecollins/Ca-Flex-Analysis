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
        
    def plot_conditions(self, data_type, combine = False, plate_number = True, activator = " ", show_window = False, dpi = 120, title = "", error = False, control = True, cmap = "winter_r", window_color = 'hotpink', proteins = [], compounds = [], **kwargs):
        """Plots each mean condition versus time, for either each plate or over all plates, for each compound and protein.
        
        If no title is desired, set title to " ".
        
        :param combine: If True, plot_conditions plots a single graph showing the mean of each condition over all plates, False plots each plate separately
        :type combine: bool
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
                
        # plots separate plots of each compound and protein, combining the data across all plates        
        if combine == True:
            data = self.data[data_type]['grouped']['data']
            time = self.data[data_type]['grouped']['time']

            group = data[data.Valid == True]
            group = group.drop('Valid', axis = 1)
            t_group = time[time.Valid == True]
            t_group = t_group.drop('Valid', axis = 1)

            # get data, time and error values for each condition
            data = group.groupby(self.grouplist)[group.columns].mean().reset_index()
            time = t_group.groupby(self.grouplist)[t_group.columns].mean().reset_index()
            yerr = group.groupby(self.grouplist)[group.columns].sem()

            # for some reason in this instance the additional columns are not dropped, so we have to do it manually
            # get columns list
            dropcols = self.grouplist.copy() + ["Contents", "Row", "Well ID", "Column"]

            # reset index
            yerr = yerr.drop(dropcols, axis = 1).reset_index()

            # get names of proteins and compounds, excluding control
            if proteins == []:
                proteins = data[data['Type'].str.contains('control') == False]['Protein'].unique()
            # iterate through proteins list
            for pkey, pval in enumerate(proteins):
                # get number of compounds for each protein
                if compounds == []:
                    compounds = data[(data['Type'].str.contains('control') == False) & (data['Protein'] == pval)]['Compound'].unique()
                # iterate through compounds for each protein
                for ckey, cval in enumerate(compounds):

                    # extract data for each protein and compound, excluding control. 
                    data_temp = data[data['Type'].str.contains('control') == False]
                    data_temp = data_temp[(data_temp['Protein'] == pval) & (data_temp['Compound'] == cval)]

                    time_temp = time[time['Type'].str.contains('control') == False]
                    time_temp = time_temp[(time_temp['Protein'] == pval) & (time_temp['Compound'] == cval)]

                    yerr_temp = yerr[yerr['Type'].str.contains('control') == False]
                    yerr_temp = yerr_temp[(yerr_temp['Protein'] == pval) & (yerr_temp['Compound'] == cval)]

                    templist = [x for x in self.grouplist if x != 'Concentration'] # get columns to remove

                    # extract just the data with conc as the index
                    index = data_temp.loc[:, :self.grouplist[-1]]
                    data_temp = data_temp.set_index('Concentration').drop(templist + ["Column"], axis = 1)
                    time_temp = time_temp.set_index('Concentration').drop(templist + ["Column"], axis = 1)
                    yerr_temp = yerr_temp.set_index('Concentration').drop(templist, axis = 1)
                    
                    if control == True:
                        control_data = data[data['Type'].str.contains('control') == True]
                        control_data = control_data[(control_data['Protein'] == pval)].drop(templist + ["Column"], axis = 1)

                        control_time = time[time['Type'].str.contains('control') == True]
                        control_time = control_time[(control_time['Protein'] == pval)].drop(templist + ["Column"], axis = 1)
                    
                    # stops empty rows being plotted
                    if (pval != 'none') & (cval != 'none'):

                        fig, ax = plt.subplots(dpi = 120)

#                         # control 
#                         if control == True:
#                             ctrl_label = "{} (control)".format(data['Contents'][(data['Type'] == 'control') & (data['Protein'] == pval)].unique()[0])
#                             ax.plot(control_time.iloc[0], control_data.iloc[0], '-o', color = 'black', mfc = 'white', lw = 1, zorder = 2, 
#                                     label = ctrl_label)

#                     plot series, iterating down rows
                        for i in range(len(time_temp)):
                            if error == True:
                                ax.errorbar(x = time_temp.iloc[i], y = data_temp.iloc[i], yerr = yerr_temp.iloc[i],
                                           capsize = 3, zorder=1, color = cal.plot_color(data_temp, cmap, i),
                                           label = "{}".format(data_temp.index[i]))
                            else: 
                                ax.plot(time_temp.iloc[i], data_temp.iloc[i], '-o', zorder=1, label = "{} {}".format(data_temp.index[i], index['Concentration Units'].iloc[i]),
                                       ms = 5, color = cal.plot_color(data_temp, cmap, i))

                            # add legend    
                            ax.legend(loc = "upper left", bbox_to_anchor = (1.0, 1.0), frameon = False, title = "{} {}".format(pval, cval))

                        # white background makes the exported figure look a lot nicer
                        fig.patch.set_facecolor('white')

                        # spines
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)

                        # add line representing the activator
                        times = self.data['baseline_corrected']['grouped']['time'].drop(['Column', 'Valid'], axis = 1).mean()
                        # get times
                        time_filter = times > (self.inject - 5) # mean time series that contains activator

                        # get start and end points
                        injection_start = times[time_filter].iloc[0]
                        injection_end = times[time_filter].iloc[-1]

                        # add line indicating presence of activator
                        ymax = data_temp.max().max() + data_temp.max().max()*0.1 # add a bit extra to prevent clash w/ data
                        ax.plot([injection_start, injection_end], [ymax, ymax], c = 'black')

                        # activator title
                        ax.text((injection_start+injection_end)/2, (ymax+ymax*0.05), activator, ha = 'center')

                        # assay title
                        ax.set_title(title, x = 0, fontweight = '550')

                        # axes labels
                        ax.set_xlabel("time (s)")
                        ax.set_ylabel("$\mathrm{\Delta Ca^{2+} \ _i}$ (Ratio Units F340/F380)")

                        if show_window == True:
                            # x min and x max for axvspan 
                            xmin = time_temp.loc[:, self.window[0]].mean()
                            xmax = time_temp.loc[:, self.window[1]].mean()
                            ax.axvspan(xmin, xmax, facecolor = window_color, alpha = 0.5)

                    plt.show()

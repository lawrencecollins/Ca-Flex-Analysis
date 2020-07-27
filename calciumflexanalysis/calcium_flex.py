import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import string, math
from platemapping import plate_map as pm
import matplotlib.patches as mpl_patches
from scipy.optimize import curve_fit

# define well plate dimensions
wells = {6:(2, 3), 12:(3, 4), 24:(4, 6), 48:(6, 8), 96:(8, 12), 384:(16, 24)} 

def read_in(raw_data):
    """Returns a dataframe of the old flex data."""
    df = pd.read_csv(raw_data, delimiter='\t', skiprows = 2, skipfooter=3, engine = 'python', encoding = 'mbcs') # update to check headers if there is no title?
    return df

def read_in_new(raw_data):
    """Returns a dataframe of the new flex data."""
    df = pd.read_csv(raw_data, delimiter='\t', skiprows = 2, skipfooter=3, engine = 'python', encoding = "utf-16", 
                    skip_blank_lines=True) 
    return df

# curve fitting functions
def _ec50_func(x,top,bottom, ec50, hill):
    z=(ec50/x)**hill
    return (bottom + ((top-bottom)/(1+z)))   

def _ic50_func(x,top,bottom, ic50, hill):
    z=(ic50/x)**hill
    return (top - ((top-bottom)/(1+z)))
func_dict = {"ic50":_ic50_func, "ec50": _ec50_func} # dicitonary to access required curve fitting function


def plot_color(dataframe, cmap, iterrange):
    """Returns a color for each index value (accessed by iterrange) of a dataframe."""
    types = list(dataframe.index)
    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0, 1, len(types)))
    colordict = dict(zip(types, colors))
    color = colordict.get(dataframe.index[iterrange])
    return color

def get_color(condition_list, cmap, name):
    """Returns a color for each condition plotted. 
    
    Similar to plot_color() but used for lists instead of dataframes - see plot_curve(combine = True)."""
    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0, 1, len(condition_list)))
    colordict = dict(zip(condition_list, colors))
    color = colordict.get(name)
    return color

class CaFlexPlate:
    """Class used for the analysis of individual Calcium Flex well plates.
    
    :param raw_data: Raw, unprocessed data from experiment
    :type raw_data: .txt file
    :param plate_map_file: Filled template plate map that contains the information for each well of the well plate
    :type plate_map_file: .csv
    :param inject: Activator injection time point
    :type inject: float or int
    :param map_type: 'short' or 'long' - Denotes the type of plate map file used, default = 'short'
    :type map_type: str
    :param size: Size of well plate - 6, 12, 24, 48, 96 or 384. plate_map_file MUST have the appropriate dimensions, default = 96
    :type size: int
    :param data_type: 'new' or 'old' - denotes type of flex data, default = 'old'
    :type data_type: str
    :param valid: Validates every well - 'True' sets every well as valid, 'False' wells will not be used for analysis, default = True
    :type valid: bool
    :param processed_data: Dictionary containing separate dataframes of the time and flex data for every well
    :type processed_data: dictionary of pandas dataframes
    :param plate_map: plate_map_file converted as a dataframe
    :type plate_map: pandas dataframe
    :param title: Title of plate or assay 
    :type title: str
    """
    def __init__(self, raw_data, plate_map_file, inject, map_type = 'short', data_type = 'old', valid = True, size = 96, title = ""):
        self.raw_data = raw_data
        self.plate_map_file = plate_map_file
        self.inject = inject
        self.map_type = map_type
        self.size = size
        self.data_type = data_type
        self.valid = valid # False invalidates all wells of plate
        self.processed_data = {'ratio':self._data_processed()}
        self.plate_map = self._give_platemap()
        self.grouplist = ['Protein','Type', 'Compound','Concentration', 'Concentration Units']
        
        # invalidate all wells if valid = False
        
        if self.valid == False:
            self.plate_map['Valid'] = valid
        if self.valid == True:
            self.plate_map['Valid'] = valid # ?????!!!!!!!!!!!??????????!!!!!!
            
        # optional title param
        self.title = title
        # add default title
        if len(title) == 0:
            self.title = raw_data[:-4] # change to experiment title in plate map?
        
        
    def _give_platemap(self):
        """Returns platemap dataframe."""
        if self.map_type == 'short':
             platemap = pm.short_map(self.plate_map_file, size = self.size, valid = self.valid)
        elif self.map_type == 'long':
            platemap = pm.plate_map(self.plate_map_file, size = self.size, valid = self.valid)
        return platemap
    
    def _data_processed(self):
        """Returns a timemap and datamap as a tuple."""
        if self.data_type == 'old':
            df = read_in(self.raw_data)
            # create new dataframe containing all time values for each well
            dftime = df.filter(regex = 'T$', axis = 1)
           # edit header names (this will come in handy in a second)
            dftime.columns = dftime.columns.str.replace('T', "")
            # extract list of header names 
            wellslist = list(dftime.columns.values)
            # transpose x and y axes of dataframe - generate time 'rows'
            dftime = dftime.transpose()
            # create new dataframe containing data measurements for each cell
            dfdata = df[wellslist]
            # transpose x and y axes
            dfdata = dfdata.transpose()
            # return timemap and datamap as a tuple
            return {'time':dftime, 'data':dfdata}
        
        if self.data_type == 'new':
            
            newdata = read_in_new(self.raw_data)
            # split the dataframe into the two data series
            data1 = newdata.iloc[:int(newdata.shape[0]/2), :]
            data1 = data1.reset_index(drop=True)
            data2 = newdata.iloc[int(newdata.shape[0]/2):, :]
            data2 = data2.reset_index(drop=True)
            newdatadict = {"data1":data1, "data2":data2}
            
            # initiate empty dictionary to house preprocessed data and time values
            preprocessed = {}
            # loop produces two dictionaries containing preprocessed flux data and their corresponding time values
            for key, value in newdatadict.items():
                # reset indexes
                value = value.reset_index(drop=True)
                dftime = value.filter(regex = 'T$', axis = 1)
                # edit header names (this will come in handy in a second)
                dftime.columns = dftime.columns.str.replace('T', "")
                # extract list of header names 
                wellslist = list(dftime.columns.values)
                # transpose x and y axes of dataframe - generate time 'rows'
                dftime = dftime.transpose()
                # create new dataframe containing data measurements for each cell
                dfdata = value[wellslist]
                # transpose x and y axes
                dfdata = dfdata.transpose()
                # return timemap and datamap as a tuple
                tempdict = {'time':dftime, 'data':dfdata}
                # append dictionary
                preprocessed[key] = tempdict
            
            # take means of the time in new dataframe
            mean_time = pd.concat((preprocessed["data1"]['time'], preprocessed["data2"]["time"]))
            mean_time = mean_time.groupby(mean_time.index).mean()
            mean_time = mean_time.reindex(wellslist)
            
            # take difference of data to get change in flux
            difference = preprocessed["data1"]['data'].divide(preprocessed["data2"]["data"])
            return {'time':mean_time, 'data':difference}
            
    def visualise_assay(self, share_y, export = False, title = "", colormap = 'Dark2_r',
             colorby = 'Type', labelby = 'Type', dpi = 200):
        """Returns color-coded and labelled plots of the data collected for each well of the well plate.
        
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
        self.title = title
        
        pm.visualise_all_series(x = self.processed_data['ratio']['time'], y = self.processed_data['ratio']['data'], 
                            share_y = share_y, platemap = self.plate_map, size = self.size, 
                            export = export, colormap = colormap,
                            colorby = colorby, labelby = labelby, 
                            dpi = dpi, title = self.title)
        
        plt.suptitle(self.title, y = 0.95)
        
    def see_plate(self, title = "", export = False, colormap = 'Paired',
             colorby = 'Type', labelby = 'Type', dpi = 150):
        """Returns a visual representation of the plate map.
    
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
        pm.visualise(self.plate_map, title = title, size = self.size, export = export, colormap = colormap,
             colorby = colorby, labelby = labelby, dpi = dpi)
    
       
    def see_wells(self, to_plot, share_y = True, colorby = 'Type', labelby = 'Type', colormap = 'Dark2_r'):
        """Returns plotted data from stipulated wells.
        :param size: Size of platemap, 6, 12, 24, 48, 96 or 384, default = 96
        :type size: int   
        :param to_plot: Wells to plot
        :type to_plot: string or list of strings (well ID's), e.g. "A1", "A2", "A3"
        :param share_y: 'True' sets y axis the same for all plots, default = 'True'
        :type share_y: bool
        :param colormap: Sets the colormap for the color-coding, optional
        :type colormap: str
        :param colorby: Chooses the parameter to color code by, for example 'Type', 'Contents', 'Concentration', 'Compound', 'Protein', 'Concentration Units', default = 'Type'
        :type colorby: str
        :param labelby: Chooses the parameter to label code by, for example 'Type', 'Contents', 'Concentration', 'Compound', 'Protein', 'Concentration Units', default = 'Type'
        :type labelby: str
        :return: Plotted data for the stipulated wells of the well plate  
        :rtype: figure
        """
        fig, axs = plt.subplots(len(to_plot), 1, figsize = (2*len(to_plot), 4*len(to_plot)), constrained_layout = True, sharey = share_y)

        for i in range(len(to_plot)):

            axs[i].plot(self.processed_data['ratio']['time'].loc[to_plot[i]], self.processed_data['ratio']['data'].loc[to_plot[i]], 
                        lw = 3, color = pm.wellcolour2(self.plate_map, colorby, colormap, i, to_plot), 
                       label = pm.labelwell(self.plate_map, labelby, i))
           
        # add label for each well
            axs[i].legend(loc = 'best', frameon = True, fancybox = True)
            axs[i].set_title("{} {}".format(to_plot[i], pm.labelwell(self.plate_map, labelby, i)))
            axs[i].set_facecolor('0.95')
            axs[i].set_xlabel("time / s")
            axs[i].set_ylabel("$\mathrm{\Delta Ca^{2+} \ _i}$ (Ratio Units F340/F380)")
        title = fig.suptitle('Flex data versus time for the wells {}'.format(', '.join(to_plot)), y = 1.05, size = '20')
        plt.show()
    
    def invalidate_wells(self, wells):
        """Invalidates specified wells and updates plate_map
        
        :param wells: Wells to invalidate
        :type wells: list of strings, e.g. ("A1", "A2", "A3")
        """
        
        self.plate_map = pm.invalidate_wells(self.plate_map, wells = wells, valid = False)
        
        
    def invalidate_rows(self, rows):
        """Invalidates specified rows and updates plate_map
        
        :param wells: Rows to invalidate
        :type wells: list of strings, e.g. ("A", "B", "C")
        """
        platemap = pm.invalidate_rows(self.plate_map, rows, valid = False)
        self.plate_map = platemap
    
    
    def invalidate_cols(self, cols):
        """Invalidates specified wells and updates plate_map
        
        :param wells: Wells to invalidate
        :type wells: list of ints, e.g. (1, 2, 3)
        """
        platemap = pm.invalidate_cols(self.plate_map, cols, valid = False)
        self.plate_map = platemap
        
        
    def baseline_correct(self):
        """Baseline corrects 'ratio' data using the pre-injection time points."""
        time_cut = self.inject - 5
        data_source = self.processed_data['ratio']
        # convert to numpy arrays
        time = data_source['time'].to_numpy()
        data = data_source['data'].to_numpy()
        # create mask from mean time values
        time_filter = np.nanmean(time,axis=0)<time_cut
        # # average over these times
        baseline = np.mean(data[:,time_filter],axis=1)
        # add dimension to enable broadcasting
        baseline = np.expand_dims(baseline, axis=1)
        # rewrite values back to dataframes
        self.processed_data['baseline_corrected'] = {}
        data_source = self.processed_data['baseline_corrected']['data'] = pd.DataFrame(data-baseline, index = data_source['data'].index)
        data_source = self.processed_data['baseline_corrected']['time'] = data_source = self.processed_data['ratio']['time']
        
        
    def get_gradients(self, data_type):
        """Returns the mean gradient over every 10 time points post injection.
        
        :param data_type: Data series to calculate plateau
        :type data_type: str
        :return: Dict containing a tuple corresponding to the start and end index and the corresponding mean gradient
        :rtype: dict[tuple, float]
        """        
        # filter for valid wells
        valid_filter = self.plate_map.Valid == True

        # add opposite time filter to extract data after injection
        time_cut = self.inject + 5
        data_source = self.processed_data[data_type]

        # convert to numpy arrays
        time = data_source['time'][valid_filter].to_numpy()
        data = data_source['data'][valid_filter].to_numpy()
        # create mask from mean time values
        post_inject_filter = np.nanmean(time,axis=0) > time_cut

        # get absolute gradient for each well along series
        gradient = abs(np.gradient(data[:, post_inject_filter], axis = 1))

        gradient_dict = {}

        index = np.array(list(data_source['data'].columns))[post_inject_filter]

        # mean gradient every ten measurements
        for i in range(gradient.shape[1]-9):

            # average of average gradients for every ten measurements post injection
            mean_gradient = np.nanmean(np.mean(gradient[:, i:(i+10)], axis=1), axis = 0)
            gradient_dict[(index[i]), (index[i]+10)] = mean_gradient
            
        return gradient_dict
    
    def get_window(self, data_type):
        """Updates self.window with the lowest mean gradient for each ten time point window post injection.
        
        :param data_type: Data series to calculate plateau
        :type data_type: str
        """
        gradient_dict = self.get_gradients(data_type)
        
        # get minimum gradient index window
        min_gradient = (min(gradient_dict, key = gradient_dict.get))

        self.window = min_gradient
        
    def def_window(self, time, data_type):
        """Manually set the plateau window.
        
        :param time: Time point at start of window
        :type time: int
        :param data_type: Data to set window on, either 'ratio' or 'baseline_corrected'
        :type data_type: str
        :return: Tuple of start and end index of plateau window
        :rtype: (int, int)
        """
        valid_filter = self.plate_map.Valid == True
        data_source = self.processed_data[data_type]
        time_df = data_source['time'][valid_filter]
        
        # create mask from mean time values
        window_filter = np.nanmean(time_df,axis=0) >= time
        index = np.array(list(data_source['data'].columns))[window_filter]
        self.window =  (index[0], index[10])
        
    def plot_conditions(self, data_type, activator = " ", show_window = False, dpi = 120, title = " ", error = False, control = True, cmap = "winter_r", window_color = 'hotpink', proteins = [], compounds = [], marker = '-o'):
        """Plots each mean condition versus time.
        
        :param data_type: Data to be plotted, either 'ratio' or 'baseline_corrected'
        :type data_type: str
        :param activator: Activator injected into assay, default = ""
        :type activator: str
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
        :return: Figure displaying each mean condition versus time
        :rtype: fig
        """

        platemap = self.plate_map
        grouplist = self.grouplist
        groupdct = {}

        for key, val in self.processed_data[data_type].items():
            data_length = val.shape[1]
            mapped = platemap.fillna('none').join(val)
            group = mapped[mapped.Valid == True].groupby(grouplist)[val.columns]
            # update dictionary
            groupdct[key] = group

        # get data, time and error values for each condition
        data = groupdct['data'].mean().reset_index()
        time = groupdct['time'].mean().reset_index()
        yerr = groupdct['data'].sem().reset_index()

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

                templist = [x for x in grouplist if x != 'Concentration'] # get columns to remove

                # extract just the data with conc as the index
                index = data_temp.iloc[:, :-data_length]
                data_temp = data_temp.set_index('Concentration').iloc[:, -data_length:]
                time_temp = time_temp.set_index('Concentration').iloc[:, -data_length:]
                yerr_temp = yerr_temp.set_index('Concentration').iloc[:, -data_length:]

                if control == True:
                    control_data = data[data['Type'].str.contains('control') == True]
                    control_data = control_data[(control_data['Protein'] == pval)].iloc[:, -data_length:]

                    control_time = time[time['Type'].str.contains('control') == True]
                    control_time = control_time[(control_time['Protein'] == pval)].iloc[:, -data_length:]

                # stops empty rows being plotted
                if (pval != 'none') & (cval != 'none'):

                    fig, ax = plt.subplots(dpi = 120)

                    # control 
                    if control == True:
                        ctrl_label = "{} (control)".format(platemap['Contents'][(platemap['Type'] == 'control') & (platemap['Protein'] == pval)].unique()[0])
                        ax.plot(control_time.iloc[0], control_data.iloc[0], marker, color = 'black', mfc = 'white', lw = 1, zorder = 2, 
                                label = ctrl_label)

                # plot series, iterating down rows
                    for i in range(len(time_temp)):
                        if error == True:
                            ax.errorbar(x = time_temp.iloc[i], y = data_temp.iloc[i], yerr = yerr_temp.iloc[i],
                                       capsize = 3, zorder=1, color = plot_color(data_temp, cmap, i),
                                       label = "{}".format(data_temp.index[i]))
                        else: 
                            ax.plot(time_temp.iloc[i], data_temp.iloc[i], marker, zorder=1, label = "{} {}".format(data_temp.index[i], index['Concentration Units'].iloc[i]),
                                   ms = 5, color = plot_color(data_temp, cmap, i))

                        # add legend    
                        ax.legend(loc = "upper left", bbox_to_anchor = (1.0, 1.0), frameon = False, title = "{} {}".format(pval, cval))

                    # white background makes the exported figure look a lot nicer
                    fig.patch.set_facecolor('white')

                    # spines
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)

                    # add line representing the activator
                    times = self.processed_data[data_type]['time'].mean() # get times
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

    def amplitude(self, data_type):
        """Calculates response amplitude for each condition, updates processed_data dictionary with 'plateau' and plate_map with amplitude column. 
        
        :param data_type: Data to use to calculate amplitudes, either 'ratio' or 'baseline_corrected'
        :type data_type: str

        """
        # get ampltitude for every condition
        amp = (self.processed_data[data_type]['data'].iloc[:, self.window[0]:self.window[1]]).to_numpy()
        # get mean amplitude for every condition
        amp_mean = np.mean(amp, axis = 1)
        # update processed_data with response amplitude
        self.processed_data['plateau'] = {}
        self.processed_data['plateau']['data'] = pd.DataFrame(amp_mean, index = self.processed_data['ratio']['data'].index, columns = ['Amplitude'])
        
    def mean_amplitude(self, use_normalised = False):
        """Returns mean amplitudes and error for each condition.
        
        The user must run the normalise method before attempting to get the mean amplitudes of the normalised amplitudes.
        
        :param use_normalised: If True, uses normalised amplitudes, default = False
        :type use_normalised: bool
        :return: Mean amplitudes and error for each condition
        :rtype: Pandas DataFrame
        """
        mapped = self.plate_map.fillna(-1).join(self.processed_data['plateau']['data'])
        if use_normalised == True:
            mapped = self.plate_map.fillna(-1).join(self.processed_data['plateau']['data_normed'])
        # group by grouplist and take mean amplitude for each condition
        # filter for valid wells
        group = mapped[mapped.Valid == True] 
        # drop columns which can cause errors w/ groupby operations
        group.drop(['Valid', 'Column'], axis = 1, inplace = True)
        mean_response = group.groupby(self.grouplist).mean().reset_index()
        if use_normalised == False:
            mean_response['Amplitude Error'] = group.groupby(self.grouplist).sem().reset_index().loc[:, 'Amplitude']
        else:
            mean_response['amps_normed_error'] = group.groupby(self.grouplist).sem().reset_index().loc[:, 'amps_normed']
        # drop empty rows
        mean_response.drop(mean_response[mean_response['Type'] == 'empty'].index)

        # update data dict
        self.processed_data['mean_amplitudes'] = mean_response
        
        return mean_response
    
    def normalise(self):
        """Normalises amplitudes to mean control amplitude"""
        mean_amps = self.mean_amplitude(use_normalised = False) # get mean control amplitude
        amps = self.processed_data['plateau']['data']
        control_amp = float(mean_amps[mean_amps['Type'] == 'control']['Amplitude'])
        self.processed_data['plateau']['data_normed'] = "test"
        self.processed_data['plateau']['data_normed'] = ((amps * 100) / control_amp).rename(columns = {'Amplitude':'amps_normed'})

        return self.processed_data['plateau']['data_normed']
    
    @staticmethod
    def _gen_curve_data(mean_amps, plot_func, use_normalised, n, proteins, compounds, **kwargs):
        """Generates data required for a fitted plot of IC50's or EC50's."""
        
        curve_data = {}
        # filter amps 
        amps = mean_amps[mean_amps.Type == 'compound']
        
        # get names of proteins
        if proteins == []:
            proteins = amps['Protein'].unique()
            
        # separate proteins 
        for pkey, pval in enumerate(proteins):
            # get compounds for each proteins
            if compounds == []:
                compounds = amps[amps['Protein'] == pval]['Compound'].unique()
            # get number of compounds for each protein
            for ckey, cval in enumerate(compounds):
                # filter dataframe for each compound in each protein
                temp = amps[(amps['Protein'] == pval) & (amps['Compound'] == cval)]
                # check there is only 1 conc unit
                if len(temp['Concentration Units'].unique()) > 1:
                    raise ValueError["One unit per condition please!"]
                    # check there is an adequate number of concs
                if len(temp['Concentration']) < n:
                    raise ValueError("Not enough concs! You've only got {} for {}, compound {}. You really need at least {} to do a fit.".format(len(temp['Concentration']), pval, cval, n))

                # get x, y and error values, c50 units, compound and protein names to use for plot
                x = temp['Concentration']
                y = temp.iloc[:, -2]
                yerr = temp.iloc[:, -1]
                c50units = temp['Concentration Units'].unique()[0]

                # get popt values for logistic regression
                popt, pcov = curve_fit(func_dict[plot_func], x, y, **kwargs) 

                # update curve_data dict
                curve_data["{}_{}".format(pval, cval)] = {'x':x, 'y':y, 'yerr':yerr, 'c50units':c50units, 'compound':cval, 
                                                          'protein':pval, 'popt':popt}
        return curve_data  
        
    def _get_curve_data(self, plot_func, use_normalised, n, proteins, compounds, **kwargs): 
        """Updates and returns self.plot_data with data required for a fitted plot of IC50's or EC50's."""
        mean_amps = self.mean_amplitude(use_normalised)
        
        curve_data = self._gen_curve_data(mean_amps, plot_func, use_normalised, n, proteins, compounds, **kwargs)
        
        self.plot_data = curve_data
        
        return self.plot_data
        
    @staticmethod
    def _plot_curve(curve_data, plot_func, use_normalised, n, proteins, compounds, error_bar, cmap, combine, activator, title, dpi, show_top_bot):
        
        legend_label = {"ic50":"IC$_{{50}}$", "ec50":"EC$_{{50}}$"}
        if combine == True:
            fig, ax = plt.subplots(dpi = dpi)
        
        for key, val in curve_data.items():
            if combine == False:
                fig, ax = plt.subplots(dpi = dpi)
            # unpack curve_data:
            x = val['x']
            y = val['y']
            yerr = val['yerr']
            c50units = val['c50units']
            popt = val['popt']
            protein = val['protein']
            compound = val['compound']
            tb_str = ""
            if show_top_bot == True:
                tb_str = "\nTop = {:.2f}\nBottom = {:.2f}".format(popt[0], popt[1])
            
            # get x and y fits
            fit_x = np.logspace(np.log10(x.min())-0.5, np.log10(x.max())+0.5, 300)
            fit = func_dict[plot_func](fit_x, *popt)
            ############################# types of plot: combine True OR False ####################
            # TRUE
            if combine == True:
                # get label for legend
                label = r"$\bf{}\ {}$".format(protein, compound) +"\n{} = {:.2f} {}\nHill Slope = {:.2f}".format(legend_label[plot_func], popt[2], c50units, popt[3])+tb_str

                # plot fit
                ax.plot(fit_x, fit, lw = 1.2, label = label, color = get_color(list(curve_data.keys()), cmap, "{}_{}".format(protein, compound)))

                # plot errors
                if error_bar == True:
                    ax.errorbar(x, y, yerr, fmt='ko',capsize=3,ms=3, color = get_color(list(curve_data.keys()), cmap, "{}_{}".format(protein, compound)))
            # FALSE        
            if combine == False:
                # get label for legend
                label = "{} = {:.2f} {}\nHill Slope = {:.2f} ".format(legend_label[plot_func], popt[2], c50units, popt[3])+tb_str

                # plot fit
                ax.plot(fit_x, fit, lw = 1.2, label = label, color = 'black')

                # plot errors
                if error_bar == True:
                    ax.errorbar(x, y, yerr, fmt='ko',capsize=3,ms=3, color = 'black')

                # add legend
                ax.legend(loc = 'best', frameon = False,framealpha=0.7, handlelength=0, handletextpad=0)

            ############# end of combine differences   ############# 
            ############# in loop modifications for both types of plot (combine = True OR False) #############   
            plt.xscale('log')    
                
            # spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            # labels
            ax.set_xlabel("[{}]".format(compound))
            ax.set_title(title, x = 0, fontweight = '550')
            
            if use_normalised == False:
                ax.set_ylabel("$\mathrm{\Delta Ca^{2+} \ _i}$ (Ratio Units F340/F380)")
            else:
                if activator == "":
                    ax.set_ylabel("% of activation")
                else:
                    ax.set_ylabel("% of activation by {}".format(activator))
            
            # title
            ax.set_title(title, x = 0, fontweight = '550')
            
            plt.minorticks_off()
        ###################################### end of plotting loop ##########################        
        # post loop mods
        
        
        # using plt legend allows use of loc = 'best' to prevent annotation clashing with line
        if combine == True:
            leg = ax.legend(loc = 'best', frameon = False,framealpha=0.7)
        
        plt.show()
        
        
    def plot_curve(self, plot_func, use_normalised = False, n = 5, proteins = [], compounds = [], error_bar = True, cmap = "Dark2", combine = False, activator = " ", title = " ", dpi = 120, show_top_bot = False, **kwargs):
        """Plots fitted curve using logistic regression with errors and IC50/EC50 values.
        
        :param plot_func: Plot function to use, either ic50 or ec50
        :type plot_func: str
        :param use_normalised: If True, uses normalised amplitudes, default = False
        :type use_normalised: bool
        :param n: Number of concentrations required for plot
        :type n: int
        :param proteins: Proteins to plot, defaults to all
        :type proteins: list
        :param compounds: Compounds to plot, defaults to all
        :type compounds: list
        :param error_bar: If True, reveals error bars, default = True
        :type error_bar = bool
        :param cmap: Color map that sets each line colour in a combined plot, default = "Dark2"
        :type cmap: bool
        :param activator: Activator injected into assay, default = " "
        :type activator: str
        :param title: Choose between automatic title or insert string to use, default = " "
        :type title: str
        :param dpi: Size of figure
        :type dpi: int
        :param show_top_bot: 'True' shows the top and bottom curve fitting values in the legend
        :type show_top_bot: bool
        :param **kwargs: Additional curve fitting arguments
        """         
        curve_data =  self._get_curve_data(plot_func, use_normalised, n, proteins, compounds, **kwargs)
        
        # do plots
        self._plot_curve(curve_data, plot_func, use_normalised, n, proteins, compounds, error_bar, cmap, combine, activator, title, dpi, show_top_bot)
        
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import string, math
from platemapping import plate_map as pm
import matplotlib.patches as mpl_patches
from scipy.optimize import curve_fit

wells = {6:(2, 3), 12:(3, 4), 24:(4, 6), 48:(6, 8), 96:(8, 12), 384:(16, 24)} 

def read_in(raw_data):
    """Returns a dataframe of the old flex data."""
    df = pd.read_csv(raw_data, delimiter='\t', skiprows = 2, skipfooter=3, engine = 'python', encoding = 'mbcs') 
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

class CaFlexAnalysis:
    """Class used for the analysis of Calcium Flex assays.
    
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
    """
    def __init__(self, raw_data, plate_map_file, inject, map_type = 'short', data_type = 'old', valid = True, size = 96):
        self.raw_data = raw_data
        self.plate_map_file = plate_map_file
        self.inject = inject
        self.map_type = map_type
        self.size = size
        self.data_type = data_type
        self.valid = valid
        self.processed_data = {'ratio':self._data_processed()}
        self.plate_map = self._give_platemap()
        self.grouplist = ['Protein','Type', 'Compound','Concentration', 'Concentration Units']
        
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
        CaFlexAnalysis.title = title
        
        pm.visualise_all_series(x = self.processed_data['ratio']['time'], y = self.processed_data['ratio']['data'], 
                            share_y = share_y, platemap = self.plate_map, size = self.size, 
                            export = export, colormap = colormap,
                            colorby = colorby, labelby = labelby, 
                            dpi = dpi, title = CaFlexAnalysis.title)
        
        plt.suptitle(CaFlexAnalysis.title, y = 0.95)
    def see_plate(self, title = "", size = 96, export = False, colormap = 'Paired',
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
        pm.visualise(self.plate_map, title = title, size = size, export = export, colormap = colormap,
             colorby = colorby, labelby = labelby, dpi = dpi)
    
       
    def see_wells(self, to_plot, share_y = True, size = 96, colorby = 'Type', labelby = 'Type', colormap = 'Dark2_r'):
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
        
        
    def get_window(self, data_type):
        """Returns the 10 time points post injection that contain the flattest average gradient.
        
        :param data_type: Data series to calculate plateau
        :type data_type: str
        """
        # calculate both baseline and ratio??
        
        # filter for valid wells
        valid_filter = self.plate_map.Valid == True

        # add opposite time filter to extract data after injection
        time_cut = self.inject - 5
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

        # get minimum gradient index window
        min_gradient = (min(gradient_dict, key = gradient_dict.get))



        self.window = min_gradient
        # return tuple???
        
    def def_window(self, time, data_type):
        """Manually set the plateau window.
        
        :param time: Time point at start of window
        :type time: int
        :param data_type: Data to set window on, either 'ratio' or 'baseline_corrected'
        :type data_type: str
        :return: Tuple of start and end index of plateau window
        :rtype: tuple of ints
        """
        valid_filter = self.plate_map.Valid == True
        data_source = self.processed_data[data_type]
        time_df = data_source['time'][valid_filter]
        # create mask from mean time values
        window_filter = np.nanmean(time_df,axis=0) >= time
        index = np.array(list(data_source['data'].columns))[window_filter]
        self.window =  (index[0], index[10])
    
    
    def plot_conditions(self, data_type, show_window = False, dpi = 120):
        """Plots each mean condition versus time.
        
        'show_window' uses axvspan to visualise what section of the series is being used to calculate amplitudes. This can be defined using 'get_window' (automatically calculates flattest gradient), or 'def_window' (allows the user to manually input the time point of the plateau).
        
        :param data_type: Data to be plotted, either 'ratio' or 'baseline_corrected'
        :type data_type: str
        :param show_plateau: If 'True', shows the location of the plateau on the series, default = False 
        :type show_plateau: bool
        :param dpi: Size of figure
        :type dpi: int
        :return: Figure displaying each mean condition versus time
        :rtype: fig
        """
        platemap = self.plate_map
        data_dict = self.processed_data[data_type] # dictionary containing df's of chosen data_type
        plt.figure(dpi = dpi)
        groupdct = {}
        for key, val in data_dict.items():
            mapped = platemap.join(val)
            group = mapped[mapped.Valid == True].groupby(self.grouplist)[val.columns]
            # update dictionary
            groupdct[key] = group
           
        # mean data and time
        mean_time = groupdct['time'].mean()
        mean_data = groupdct['data'].mean()
        error_data = groupdct['data'].sem()
        # plot series for each condition and control
        for i in range(len(mean_time)):
            plt.errorbar(mean_time.iloc[i], mean_data.iloc[i], yerr=error_data.iloc[i], 
                         capsize = 3, 
                         label = "{}, {}".format(list(mean_data.index.get_level_values('Concentration'))[i], 
                                                list(mean_data.index.get_level_values('Compound'))[i])) 
            # add label function that concatenates conc w/ the correct units 
        plt.legend(loc = "upper right", bbox_to_anchor = (1.35, 1.0))
        plt.xlabel("time (s)")
        plt.ylabel("$\mathrm{\Delta Ca^{2+} \ _i}$ (Ratio Units F340/F380)")
        
        # add line representing the activator
        times = data_dict['time'].mean() # get times
        time_filter = times > (self.inject - 5) # mean time series that contains activator
        # get start and end points
        injection_start = times[time_filter].iloc[0]
        injection_end = times[time_filter].iloc[-1]
        # get max y 
        ymax = mean_data.max().max() + mean_data.max().max()*0.1 # add a bit extra to prevent clash w/ data
        plt.plot([injection_start, injection_end], [ymax, ymax], c = 'black')
        
        # show window
        if show_window == True:
            # x min and x max for axvspan 
            xmin = data_dict['time'].loc[:, self.window[0]].mean()
            xmax = data_dict['time'].loc[:, self.window[1]].mean()
            plt.axvspan(xmin, xmax, facecolor = 'hotpink', alpha = 0.5)
        
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
        
    def mean_amplitude(self):
        """Returns mean amplitudes and error for each condition.
        
        :return: Mean amplitudes and error for each condition
        :rtype: Pandas DataFrame
        """
        mapped = self.plate_map.fillna(-1).join(self.processed_data['plateau']['data'])
        # group by grouplist and take mean amplitude for each condition
        # filter for valid wells
        group = mapped[mapped.Valid == True] 
        # drop columns which can cause errors w/ groupby operations
        group.drop(['Valid', 'Column'], axis = 1, inplace = True)
        mean_response = group.groupby(self.grouplist).mean().reset_index()
        mean_response['Error'] = group.groupby(self.grouplist).sem().reset_index()['Amplitude']
        return mean_response
    
    # define plotting function (cleans up plot_curve)
    def _logistic_regression(self, x, y, yerr, plot_func, compound, protein, c50units, dpi, **kwargs):
        """Plots logistic regression fit with errors on y axis. """
        # c50 dictionary for accessing functions for plot fitting
        func_dict = {"ic50":_ic50_func, "ec50": _ec50_func}
        
        # get popt values for logistic regression
        popt, pcov = curve_fit(func_dict[plot_func], x, y, **kwargs) 
        
        # x values for fit
        fit_x = np.logspace(np.log10(x.min())-0.5, np.log10(x.max())+0.5, 300) # extend by half an order of mag
        # fit y values
        fit = func_dict[plot_func](fit_x, *popt)

        # annotations
        legend_label = {"ic50":"IC$_{{50}}$", "ec50":"EC$_{{50}}$"}
        l = ["{} = {:.2f} {} \nHill slope = {:.2f}".format(legend_label[plot_func], popt[2], c50units, popt[3])]

        # initialise figure - CHANGE TO SUBPLOTS 
        fig = plt.figure(dpi = dpi)

        # plot line of best fit
        plt.plot(fit_x, fit, c = 'black', lw = 1.2)
        # plot errors (for each mean condition)
        plt.errorbar(x, y, yerr, fmt='ko',capsize=3,ms=3, c = 'black', label = l)

        # generate empty handle (hides legend handle - see next comment)
        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", 
                                         lw=0, alpha=0)]
        # using plt legend allows use of loc = 'best' to prevent annotation clashing with line
        leg = plt.legend(handles, l, loc = 'best', frameon = False,framealpha=0.7, 
                  handlelength=0, handletextpad=0)

        # log x scale (long conc)
        plt.xscale('log')
        plt.minorticks_off()

        # axes labels
        plt.ylabel("$\mathrm{\Delta Ca^{2+} \ _i}$ (Ratio Units F340/F380)")
        plt.xlabel("[{}]".format(compound))
        plt.title(protein)
        
    def plot_curve(self, plot_func, type_to_plot = 'compound', title = 'auto', dpi = 120, n = 5, **kwargs):
        """Plots fitted curve using logistic regression with errors and IC50/EC50 values.
        
        :param plot_func: Plot function to use, either ic50 or ec50
        :type plot_func: str
        :param type_to_plot: Type of condition to plot, default = 'compound'
        :type type_to_plot: str
        :param title: Choose between automatic title or insert string to use, default = 'auto'
        :type title: str
        :param dpi: Size of figure
        :type dpi: int
        :param n: Number of concentrations required for plot
        :type n: int
        :return: Figure with fitted dose-response curve
        :rtype: fig
        """
        # get data 
        table = self.mean_amplitude()
        amps = self.mean_amplitude()[self.mean_amplitude().Type == 'compound']
        
        # get names of proteins and compounds
        proteins = amps['Protein'].unique()
        compounds = amps['Compound'].unique()
        
        # get number of proteins and compounds
        p_len = len(proteins)
        c_len = len(compounds)
        
        # check units and number of concentrations
        try:
            # seperate proteins
            for i in range(p_len):
                # seperate compounds for each protein
                for j in range(c_len):
                    # filter dataframe for each compound in each protein
                    temp = amps[(amps['Protein'] == proteins[i]) & (amps['Compound'] == compounds[j])]
                    # check there is only 1 conc unit
                    if len(temp['Concentration Units'].unique()) > 1:
                        raise ValueError["One unit per condition please!"]
                        # check there is an adequate number of concs
                    if len(temp['Concentration']) < n:
                        raise ValueError("Not enough concs! You've only got {} for {}, compound {}. You really need at least {} to do a fit.".format(len(temp['Concentration']), proteins[i], compounds[j], n))

                    # get x, y and error values, c50 units, compound and protein names to use for plot
                    x = temp['Concentration']
                    y = temp['Amplitude']
                    yerr = temp['Error']
                    c50units = temp['Concentration Units'].unique()[0]
                    compound = compounds[j]
                    protein = proteins[i]
                    
                    # plot curve with line of best fit
                    self._logistic_regression(x, y, yerr, plot_func, compound, protein, c50units, dpi)
                    
        except:
            print("value error exception")
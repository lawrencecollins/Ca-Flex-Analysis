import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import string, math
from platemapping import plate_map as pm

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

class CaFlexAnalysis:
    """Class used for the analysis of Calcium Flex assays.
    
    :param raw_data: Raw, unprocessed data from experiment
    :type raw_data: .txt file
    :param plate_map_file: Filled template plate map that contains the information for each well of the well plate
    :type plate_map_file: .csv
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
    
    def __init__(self, raw_data, plate_map_file, map_type = 'short', data_type = 'old', valid = True, size = 96):
        self.raw_data = raw_data
        self.plate_map_file = plate_map_file
        self.map_type = map_type
        self.size = size
        self.data_type = data_type
        self.valid = valid
        self.processed_data = {'ratio':self._data_processed()}
        self.plate_map = self._give_platemap()
        self.processed_data['baseline_corrected'] = self._baseline_correct()
        self.plateau = self._find_plateau()
        
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
            axs[i].set_ylabel("$\mathrm{Ca^{2+} \Delta F(340/380)}$")
        title = fig.suptitle('Flex data versus time for the wells {}'.format(', '.join(to_plot)), y = 1.05, size = '20')
        plt.show()
    
    def invalidate_wells(self, wells):
        """Invalidates specified wells and updates plate_map.
        
        :param wells: Wells to invalidate
        :type wells: list of strings, e.g. ("A1", "A2", "A3")
        """
        
        self.plate_map = pm.invalidate_wells(self.plate_map, wells = wells, valid = False)
        
        
    def invalidate_rows(self, rows):
        """Invalidates specified rows and updates plate_map.
        
        :param wells: Rows to invalidate
        :type wells: list of strings, e.g. ("A", "B", "C")
        """
        platemap = pm.invalidate_rows(self.plate_map, rows, valid = False)
        self.plate_map = platemap
    
    
    def invalidate_cols(self, cols):
        """Invalidates specified wells and updates plate_map.
        
        :param wells: Wells to invalidate
        :type wells: list of ints, e.g. (1, 2, 3)
        """
        platemap = pm.invalidate_cols(self.plate_map, cols, valid = False)
        self.plate_map = platemap
        
        
    def _baseline_correct(self): 
        """Baseline correction to pre-injection data."""
        inject = 60 # ADD INJECT TO INITIATION
        time_cut = inject - 5
        data_source = self.processed_data['ratio']
        #convert to numpy arrays
        time = data_source['time'].to_numpy()
        data = data_source['data'].to_numpy()
        # create mask from mean time values
        time_filter = np.nanmean(time, axis = 0) < time_cut
        # average over these times
        baseline = np.mean(data[:, time_filter], axis = 1)
        # add dimenstion to enable broadcasting
        baseline = np.expand_dims(baseline, axis = 1)
        #rewrite values back to dataframes
        self.processed_data['baseline_corrected'] = {}
        data = pd.DataFrame(data-baseline, index = data_source['time'].index)
        time =  self.processed_data['ratio']['time']
        
        return {'time':time, 'data':data}
        
        
        
    def plot_conditions(self, data_type, show_plateau = False):
        """Plot mean of each condition with standard error bars. 
        
        :param data_type: Data type to plot, either 'ratio' or 'baseline_corrected'
        :type data_type: str
        :param show_plateau: Adds axvspan corresponding to flattest mean gradient over 10 time points post injection
        :type show_plateau: bool
        :return: Plotted mean data for each condition
        :rtype: fig
        """
        platemap = self.plate_map
        grouplist = ['Protein','Type', 'Compound','Concentration']
        groupdct = {}
        
        for key, val in self.processed_data[data_type].items():
            # join time and data to plate map 
            mapped = platemap.join(val)
            # group by protein, type, compound and concentration
            group = mapped[mapped.Valid == True].groupby(grouplist)[val.columns]
            # update dict
            groupdct[key] = group
            
        # plot series for each mean condition and control
        for i in range(len(groupdct['time'].mean())):
            plt.errorbar(groupdct['time'].mean().iloc[i], groupdct['data'].mean().iloc[i], yerr=groupdct['data'].sem().iloc[i], capsize = 3, label = "{}, {}".format(list(groupdct['data'].mean().index.get_level_values('Concentration'))[i], list(groupdct['data'].mean().index.get_level_values('Compound'))[i]))
            
            # add labels
        plt.legend(loc = "upper right", bbox_to_anchor = (1.35, 1.0))
        plt.xlabel("time / s")
        plt.ylabel("$\mathrm{Ca^{2+} \Delta F(340/380)}$") # UPDATE LABELS 
        
        # axvspan
        if show_plateau == True:
            # x min and x max for axvspan 
            xmin = self.processed_data[data_type]['time'].loc[:, self.plateau[0]].mean()
            xmax = self.processed_data[data_type]['time'].loc[:, self.plateau[1]].mean()
            plt.axvspan(xmin, xmax, facecolor = 'pink', alpha = 0.5)

        plt.show()
                
    def _find_plateau(self, data_type = 'baseline_corrected'):
        """Finds the time points corresponding to the average gradient that is closest to 0, between 10 time points.
        
        :param data_type: Data type to plot, either 'ratio' or 'baseline_corrected', default = 'baseline_corrected'
        :type data_type: str
        """
        # Filter for valid wells
        valid_filter = self.plate_map.Valid == True
        
        # add opposite time filter to extract data after injection
        inject = 60 # ADD INJECT TO INITIATION
        time_cut = inject - 5 
        data_source = self.processed_data[data_type]
        # convert to numpy arrays
        time = data_source['time'][valid_filter].to_numpy()
        data = data_source['data'][valid_filter].to_numpy()
        # create mask from mean time values
        post_inject_filter = np.nanmean(time,axis=0) > time_cut
        
        # get absolute gradient for each well along series
        gradient = abs(np.gradient(data[:,post_inject_filter], axis = 1))

        gradient_dict = {}
        index = np.array(list(data_source['data'].columns))[post_inject_filter]

        

        # mean gradient every ten measurements
        for i in range(gradient.shape[1]-10):

            # average of average gradients for every ten measurements post injection
            mean_gradient = np.nanmean(np.mean(gradient[:, i:(i+10)], axis=1), axis = 0)
            gradient_dict[(index[i]), (index[i]+10)] = mean_gradient

        # get minimum gradient index window
        min_gradient = (min(gradient_dict, key = gradient_dict.get))
        
        amp = (self.processed_data['baseline_corrected']['data'].iloc[:, min_gradient[0]:min_gradient[1]]).to_numpy()
        amp_mean = np.mean(amp, axis = 1)
        return min_gradient
        
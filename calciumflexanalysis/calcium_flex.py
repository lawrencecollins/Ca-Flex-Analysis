import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import string, math
from platemapping import plate_map as pm

wells = {6:(2, 3), 12:(3, 4), 24:(4, 6), 48:(6, 8), 96:(8, 12), 384:(16, 24)} 

def read_in(raw_data):
    """returns a dataframe of the flex data"""
    df = pd.read_csv(raw_data, delimiter='\t', skiprows = 2, skipfooter=3, engine = 'python', encoding = 'mbcs') 
    return df

class CaFlexAnalysis:
    """Class used for the analysis of Calcium Flex assays
    
    :param raw_data: Raw, unprocessed data from experiment
    :type raw_data: .txt file
    :param plate_map_file: Filled template plate map that contains the information for each well of the well plate
    :type plate_map_file: .csv
    :param map_type: 'short' or 'long' - Denotes the type of plate map file used
    :type map_type: str
    :param size: Size of well plate - 6, 12, 24, 48, 96 or 384. plate_map_file MUST have the appropriate dimensions.
    :type size: int
    :param data_type: 'new' or 'old' - denotes type of flex data
    :type data_type: str
    :param valid: Validates every well - 'True' sets every well as valid, 'False' wells will not be used for analysis, optional
    :type valid: bool
    :param processed_data: Dictionary containing separate dataframes of the time and flex data for every well
    :type processed_data: dictionary of pandas dataframes
    :param plate_map: plate_map_file converted as a dataframe
    :type plate_map: pandas dataframe"""
    
    def __init__(self, raw_data, plate_map_file, map_type = 'short', data_type = 'old', valid = True, size = 96):
        self.raw_data = raw_data
        self.plate_map_file = plate_map_file
        self.map_type = map_type
        self.size = size
        self.data_type = data_type
        self.valid = valid
        self.processed_data = self._data_processed
        self.plate_map = self._give_platemap
        
    def _give_platemap(self):
        """Returns platemap dataframe"""
        if self.map_type == 'short':
             platemap = pm.short_map(self.plate_map_file, size = self.size, valid = self.valid)
        elif self.map_type == 'long':
            platemap = pm.plate_map(self.plate_map_file, size = self.size, valid = self.valid)
        return platemap
    
    def _data_processed(self):
        """returns a timemap and datamap as a tuple"""
        platemap = self._give_platemap()
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
    
    def visualise_assay(self, export = False, title = "", colormap = 'Dark2_r',
             colorby = 'Type', labelby = 'Type', dpi = 200):
        """Returns color-coded and labelled plots of the data collected for each well of the well plate
        
        :param export: If 'True' a .png file of the figure is saved, optional
        :type export: bool
        :param title: Sets the title of the figure, optional
        :type title: str
        :param colormap: Sets the colormap for the color-coding, optional
        :type colormap: str
        :param colorby: Chooses the parameter to color code by, choose between 'Type', 'Contents', 'Concentration', 'Compound', 'Protein', 'Concentration Units', optional
        :type colorby: str
        :param labelby: Chooses the parameter to label code by, choose between 'Type', 'Contents', 'Concentration', 'Compound', 'Protein', 'Concentration Units', optional
        :type labelby: str
        :param dpi: Size of the figure, optional
        :type dpi: int
        :return: Figure of plotted data for each well of the well plate described in plate_map_file
        :rtype: figure
        """
        CaFlexAnalysis.title = title
        
        pm.visualise_all_series(x = self.processed_data()['time'], y = self.processed_data()['data'], 
                            platemap = self.plate_map(), size = self.size, 
                            export = export, colormap = colormap,
                            colorby = colorby, labelby = labelby, 
                            dpi = dpi, title = CaFlexAnalysis.title)
        
        plt.suptitle(CaFlexAnalysis.title, y = 0.95)

    def visualise_wells(self, to_plot, colorby = 'Type', labelby = 'Type', colormap = 'Dark2_r'):
        """Returns plotted data from stipulated wells
        
        :param to_plot: Wells to plot
        :type to_plot: list of strings (well ID's), e.g. "A1", "A2", "A3"
        :param colormap: Sets the colormap for the color-coding, optional
        :type colormap: str
        :param colorby: Chooses the parameter to color code by, choose between 'Type', 'Contents', 'Concentration', 'Compound', 'Protein', 'Concentration Units', optional
        :type colorby: str
        :param labelby: Chooses the parameter to label code by, choose between 'Type', 'Contents', 'Concentration', 'Compound', 'Protein', 'Concentration Units', optional
        :type labelby: str
        :return: Plotted data for the stipulated wells of the well plate  
        :rtype: figure
        """
        CaFlexAnalysis.to_plot = to_plot 
        
        pm.plot_series(x = self.processed_data()['time'], y = self.processed_data()['data'],
                      platemap = self.plate_map(), to_plot = CaFlexAnalysis.to_plot, size = self.size, 
                       colorby = colorby, labelby = labelby, colormap = colormap)
    
    def invalidate_wells(self, wells):
        """Invalidates specified wells and updates plate_map
        
        :param wells: Wells to invalidate
        :type wells: list of strings, e.g. ("A1", "A2", "A3")
        """
        self.wells = wells
        platemap = pm.invalidate_wells(self.plate_map(), wells = wells, valid = False)
        self.plate_map = platemap

        
    def invalidate_rows(self, rows):
        """Invalidates specified rows and updates plate_map
        
        :param wells: Rows to invalidate
        :type wells: list of strings, e.g. ("A", "B", "C")
        """
        self.rows = rows
        platemap = pm.invalidate_rows(self.plate_map(), rows, valid = False)
        self.plate_map = platemap
    
    def invalidate_cols(self, cols):
        """Invalidates specified wells and updates plate_map
        
        :param wells: Wells to invalidate
        :type wells: list of ints, e.g. (1, 2, 3)
        """
        self.cols = cols
        platemap = pm.invalidate_cols(self.plate_map(), cols, valid = False)
        self.plate_map = platemap

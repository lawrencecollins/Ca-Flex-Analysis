import numpy as np
from calciumflexanalysis import calcium_flex as cal
from collections import defaultdict

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
        
        # iterate through each plate and update attributes using predefined caflexanalysis methods
        for key, val in enumerate(self.caflexplates):
            # titles (nice ref for the user)
            self.titles["plate_{}".format(key+1)] = val.title
            # update processed data w/ ratios
            self.data['ratio']["plate_{}".format(key+1)] = val.processed_data['ratio']
            # dictionary of plate maps for each plate
            self.plate_maps["plate_{}".format(key+1)] = val.plate_map
    
    
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

        for key, val in enumerate(plates):
            val.baseline_correct()
    
    def get_window(self, data_type):
        """Finds the lowest overall mean gradient for across the ten time point window post injection for the plates
        
        :param data_type: Data series to calculate plateau
        :type data_type: str
        :return: Tuple of start and end index of plateau window
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
        :return: Tuple of start and end index of plateau window
        :rtype: (int, int)
        """
        plates = self.caflexplates
        for key, val in enumerate(plates):
            val.def_window(time, data_type)
            
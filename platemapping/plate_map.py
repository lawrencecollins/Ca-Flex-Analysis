import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import string, math

# we need to reference well plate dimensions  
wells = {6:(2, 3), 12:(3, 4), 24:(4, 6), 48:(6, 8), 96:(8, 12), 384:(16, 24)} # dictionary of well sizes  

# specifying column types for the eventual dataframes prevents some problems when handling them 
data_types = {'Well ID' : str, 'Compound' : str, 'Protein': str, 'Concentration' : float, 'Concentration Units' : str,
             'Contents' : str, 'Type' : str, 'Valid' : bool}

# EMPTY MAP GENERATION
def empty_map(size = 96, valid = True):
    """generates an empty platemap of defined size"""
    
    # import alphabet for row labels
    letters = list(string.ascii_uppercase)
    # define rows (note wells defined earlier)
    rows = letters[0:(wells[size])[0]]
    # list of cell letters
    cellstemp1 = rows*(wells[size])[1]
    # sorting EITHER rows or columns lists the well ID's in the correct order
    cellstemp1.sort()
    
    # define the correct number of columns according to the well plate
    columns = list(range(1, (wells[size])[1]+1))
    # list of cell numbers for every well
    cellstemp2 = columns*(wells[size])[0]
    # dictionary of cell letters (rows) and numbers (columns)
    cellsdict = {'Row':cellstemp1, 'Column':cellstemp2}
    
    # new empty dataframe to append with wells
    df = pd.DataFrame(cellsdict)
    df["Well ID"] = df["Row"] + df["Column"].astype(str)
    
    headers = ("Well ID", "Type", "Contents", "Compound", "Protein", 
               "Concentration", "Concentration Units","Row", "Column", "Valid")    
    df = df.reindex(headers, axis = "columns")
    
    # valid column allows easy ommision of anomalous data
    df['Valid'] = df['Valid'].fillna(valid)
    
    # type column provides a quick identification of what is in each well
    df['Type'] = df['Type'].fillna('empty')
    
    # seting index to Well ID provides a uniform index for all dataframes generated from a particular well plate
    df.set_index(df['Well ID'], inplace = True)
    
    return df

# PLATE DF GENERATION FROM LONG HAND MAP
def plate_map(file, size = 96, valid = True):
    """generates a dataframe from a 'long' plate map csv file that defines each and every well from a well plate of defined size"""
    # substitute values w/ new plate map
    df = pd.read_csv(file, skiprows = 1, dtype = data_types, skipinitialspace = True)

    # set index to Well ID
    df = df.set_index(df['Well ID'])
    
    # correct typos due to capitalisation and trailing spaces
    df['Type'] = df['Type'].str.lower()
    df[['Contents', 'Compound', 'Protein', 'Type']] = df[['Contents', 'Compound', 
                                                                      'Protein', 'Type']].stack().str.rstrip().unstack()
    
    # define empty plate map
    temp = empty_map(size = size, valid = valid)
    
    # insert plate map into empty map
    temp.update(df)
    
    temp.drop(['Well ID'], axis=1)
    return temp

# PLATE DF GENERATION FROM SHORT HAND MAP
def short_map(file, size = 96, valid = True):
    """generates a dataframe from a 'short' plate map csv file that defines each and every well from a well plate of defined size"""

    # read in short map 
    df = pd.read_csv(file, skiprows = 1, skipinitialspace = True)

    # generate empty dataframe to append with each duplicated row
    filleddf = pd.DataFrame()

    # iterate down rows of short map to create duplicates that correspond to every 'filled' well plate
    for i in range(len(df.index)):
        row = df.iloc[i]
        # generate temporary dataframe for each row
        temp = pd.DataFrame()
        # duplicate rows according to difference in start and end and add to temp dataframe
        temp = temp.append([row]*(row['End']-row['Start'] +1), ignore_index = True)
        # update column coordinates using index of appended dataframe
        temp['Column']= (temp['Start'])+temp.index
        # concatenate column and row coordinates to form empty well ID
        temp['ID']= temp['Row'] + temp['Column'].astype('str')
        # set index to well ID
        temp.set_index('ID', inplace = True)
        # add generated rows to new dataframe
        filleddf = filleddf.append(temp)
    
    # insert filled df into empty plate map to include empty rows 
    finalmap = empty_map(size = size, valid = valid)
    finalmap.update(filleddf)
    # update data types to prevent future problems
    finalmap['Column'] = finalmap['Column'].astype(int)
    # correct typos due to capitalisation and trailing spaces
    finalmap['Type'] = finalmap['Type'].str.lower()
    finalmap[['Contents', 'Compound', 'Protein', 'Type']] = finalmap[['Contents', 'Compound', 
                                                                      'Protein', 'Type']].stack().str.rstrip().unstack()

    
    return finalmap

# The next 3 functions are used to simplify 'visualise' function that follows: 

# hatches are defined to clearly show invalidated wells
hatchdict = {"True":"", "False":"////"}

# fontsize will scale font size of visualisaiton to the well plate size (avoids overlapping text)
def fontsize(sizeby, size): 
    """returns a font size defined by the length of the string and size of the well plate
    (larger well plate and/or longer string = smaller font size."""
    return (8 - math.log10(len(str(sizeby)))*2 - math.log10(size)*1.5)

# adds labels according to label stipulations (avoids excessive if statements in the visualise function)
def labelwell(platemap, labelby, iterrange):
    """returns label for each row of a stipulated column"""
    if platemap['Type'].iloc[iterrange] == 'empty':
        return "empty"
    else:
        return str(platemap[labelby].iloc[iterrange]).replace(" ", "\n")
    
def wellcolour(platemap, colorby, colormap, iterrange):
    """returns a unique colour for each label or defined condition"""
    # unique strings in the defined column are used as the list of labels, converted to strings to avoid errors.
    types = [str(i) for i in list(platemap[colorby].unique())]
    cmap = plt.get_cmap(colormap)
    # get equally spaced colour values
    colors = cmap(np.linspace(0, 1, len(types)))
    colordict = dict(zip(types, colors))
    colordict['nan'] = 'yellow'
    color = colordict.get(str(platemap[colorby].iloc[iterrange]))
    return color

def visualise(platemap, title = "", size = 96, export = False, colormap = 'Paired',
             colorby = 'Type', labelby = 'Type', dpi = 150):
    """returns a visual representation of the platemap"""
    fig = plt.figure(dpi = dpi)
    # define well plate grid according to size of well plate 
    # an extra row and column is added to the grid to house axes labels
    grid = gridspec.GridSpec((wells[size])[0]+1, (wells[size])[1]+1, wspace=0.1, hspace=0.1, figure = fig)

    # plot row labels in extra row
    for i in range(1, (wells[size])[0]+1):
        ax = plt.subplot(grid[i, 0])
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.text(0.5, 0.5, list(string.ascii_uppercase)[i-1], size = 10, ha = "center", va="center")
        
    # plot column labels in extra column
    for i in range(1, (wells[size])[1]+1):
        ax = plt.subplot(grid[0, i])
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.text(0.5, 0.5, list(range(1, (wells[size])[1]+1))[i-1], size = 8, ha = "center", va="center")
        
    # plot plate types in grid, color code and label
    for i in range(size):
            # color code
            ax = plt.subplot(grid[(ord(platemap['Row'].iloc[i].lower())-96), ((platemap['Column'].iloc[i]))])
            ax.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            # Well colour coding  
            if platemap['Type'].iloc[i] == 'empty':
                ax.add_artist(plt.Circle((0.5, 0.5), 0.49, edgecolor='black', fill = False, lw=0.5))
                
            else:
                ax.add_artist(plt.Circle((0.5, 0.5), 0.49, facecolor=wellcolour(platemap, colorby, colormap, i), 
                                          edgecolor='black', lw=0.5, hatch = hatchdict[str(platemap['Valid'].iloc[i])]))
            
            # LABELS
            ax = fig.add_subplot(grid[(ord(platemap['Row'].iloc[i].lower())-96), ((platemap['Column'].iloc[i]))])
            
            # nan option allows a blank label if there is nothing stipulated for this label condition
            if str(platemap[labelby].iloc[i]) == 'nan':
                pass
            else:
                ax.text(0.5, 0.5, labelwell(platemap, labelby, i), 
                        size = str(fontsize(sizeby = platemap[labelby].iloc[i], size = size)), 
                        wrap = True, ha = "center", va="center")
    # add title 
    fig.suptitle('{}'.format(title))
    
    # provides option to save well plate figure 
    if export == True:
        plt.savefig('{}_map.png'.format(title))
    else:
        pass


# Puts the assay data into a suitable form to aid future analysis
def readandmap(data, platemap, size = 96):
    """returns a timemap and datamap as a tuple"""
    # create dataframe from .txt file 
    def read_in(data):
        df = pd.read_csv(data, delimiter='\t', skiprows = 2, skipfooter=3, engine = 'python', encoding = 'mbcs') 
        return df
    # read in assay data 
    df = read_in(data)
    # create new dataframe containing all time values for each well
    dftime = df.filter(regex = 'T$', axis = 1)
    # edit header names (this will come in handy in a second)
    dftime.columns = dftime.columns.str.replace('T', "")
    # extract list of header names 
    wellslist = list(dftime.columns.values)
    # transpose x and y axes of dataframe - generate time 'rows'
    dftime = dftime.transpose()
    # join time rows to plate map, generating plate map that contains time values
    timemap = platemap.join(dftime)

    # create new dataframe containing data measurements for each cell
    dfdata = df[wellslist]
    # transpose x and y axes
    dfdata = dfdata.transpose()
    # join to plate map 
    datamap = platemap.join(dfdata)
    
    # return timemap and datamap as a tuple
    return timemap, datamap


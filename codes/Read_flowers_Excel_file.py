# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 09:23:01 2020

Date of last modification: 14/06/2022

@author: Unité Imagerie - Myriam Oger


Read_flowers_Excel_file.py
Open an Excel file and put all the infos concerning each flower inside a dictionary

Usage:
    Read_flowers_Excel_file.py
    Read_flowers_Excel_file.py (-h | --help)
    Read_flowers_Excel_file.py <file> [--typef=<typef>] [--num_lot=<numlot>] \
    [--vol_nectar=<voln>]

Options:  
    -h, --help              for help on syntax
    -t, --typef TYPEF       to take the infos only for one type of flower. The\
    different possibilities are: male/female/hermaphrodite/all [default: 'all']
    -l, --num_lot NUMLOT    to take only one batch: 1 à 8 / all [default: 'all']
    -v, --vol_nectar VOLN   to select the flower where the volume of nectar\
    collected is known: yes/no/all [defaut: 'all']
"""


#~~ Import of the libraries ~~
import xlrd
import os
import sys
from docopt import docopt
from tkinter.filedialog import askopenfilename

def name_file(dirfile):
    """ Verify if the filename given for the Excel file is valid. If it's not, 
    a dialog box is shown to choose an Excel file.
    in: dirfile     = a path
    out:- dirfile   = the valid path
        - directory = the directory where the Excel file can be found
        - filexls   = the name of the Excel file
    """
    # If the name of the Excel file is not given correctly, a dialog box is shown
    if not dirfile or not dirfile.find('.xls') or not os.path.isfile(dirfile):
        FILETYPES = [("excel files", "*.xlsx"),("All", "*")]
        dirfile = askopenfilename(title='Select an Excel file', filetypes=FILETYPES)

    # Name of the directory/folder
    directory = os.path.dirname(dirfile)
    # Name of the Excel file
    filexls = os.path.basename(dirfile)
    return [dirfile, directory, filexls]

def sort_dico(dicotemp, typef=None, num_batch=None, vol_nectar=None):
    """ 
    Verify if the current flower follow the requirement given via command line\
    arguments.
    in:     - dicotemp      = the temporary dictionary
            - typef         = type of flowers wanted (False = all)
            - num_batch     = the number of the batch wanted (False = all)
            - vol_nectar    = if the volume of nectar is required ()
    out: True = the current flower is kept
        False = the current flower is rejected
    """
    if typef and not(dicotemp['Flower type'].find(typef)):
        return False
    if num_batch and dicotemp['# batch'] != num_batch:
        return False
    if vol_nectar == True and dicotemp['Nectar volume (µl)'] == 'NC':
        return False
    if vol_nectar == False and dicotemp['Nectar volume (µl)'] != 'NC':
        return False
    return True
    

def Dico_images(file, typef=None, num_lot=None, vol_nectar=None):
    """
    Read the Excel file and return a dictionary containing the flowers that \
    follow the requirements.
    in:     - file          = an Excel file
            - typef         = type of flowers wanted
            - num_batch     = the number of the batch wanted
            - vol_nectar    = if the volume of nectar is required
    out: dico = the filtered dictionary. Dictionary keys correspond to the\
    names given during the scan of the flowers.
    """
    dirfile, directory, filexls = name_file(file)
    wb = xlrd.open_workbook(dirfile)
    
    sh = wb.sheet_by_name(wb.sheet_names()[0])
    
    num_rows = sh.nrows
    num_cols = sh.ncols
    
    #print(f'num_rows = {num_rows} et num_cols = {num_cols}')
    curr_row = 0
    l_keys = []
    for cell in range(num_cols):
        l_keys.append(sh.cell_value(curr_row, cell))
    
    curr_row += 1
    
    dico = {}
    while curr_row < num_rows:
        dic_temp = {}
        curr_col = 1
        while curr_col<num_cols:
            cell_value = sh.cell_value(curr_row, curr_col)
            dic_temp[l_keys[curr_col]] = cell_value
            curr_col += 1
        if sort_dico(dic_temp, typef, num_lot, vol_nectar):
            dico[dic_temp['ID_Imagerie']] = dic_temp
        curr_row += 1
    return dico

def main(argv):
    """Fonction principale"""
    # Initialisation of variables
    repfic = None
    typef = None
    num_lot = None
    vol_nectar = None
    
    # Evaluation of variables given as arguments via command lines
    for key, val in argv.items():
        if key in ("-h", "--help"):
            if val:
                sys.exit(2)
        elif key in ("-t", "--typef"):
            typef = eval(val)
            if typef == 'all':
                typef = None
            elif typef in ['male','Male','MALE']:
                typef = 'male'
            elif typef in ['female', 'Female', 'FEMALE']:
                typef = 'female'
            elif typef in ['hermaphrodite', 'Hermaphrodite', 'HERMAPHRODITE']:
                typef = 'hermaphro'
            else:
                print("Unknown flower type")
                sys.exit()
        elif key in ("-l", "--num_batch"):
            num_batch = eval(val)
            if num_batch =='all':
                num_batch = None
        elif key in ("-v", "--vol_nectar"):
            if val == None:
               vol_nectar = None
            else:
                vol_nectar = eval(val)
            if vol_nectar =='all':
                vol_nectar = None
        elif key in ('<file>',):
            repfic = val
    
    dico_img = Dico_images(repfic, typef, num_lot, vol_nectar)
    print(dico_img)


if __name__ == "__main__":
    # variable containing au the options given as argument in the command line
    arguments = docopt(__doc__)
    main(arguments)
    
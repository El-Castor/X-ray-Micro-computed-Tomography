# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 14:04:50 2020

Date de dernière modification : 14/06/2022

@author: Unité Imagerie - Myriam Oger


Segmentation_nectary

Usage:
    Segmentation_nectary.py
    Segmentation_nectary.py (-h | --help)
    Segmentation_nectary.py <name_directory> [--seuil=<seuil>]
    Segmentation_nectary.py [--path=<path>] [--type=<type>] [--hide | --show]\
    [--quiet | --verbose] [--seuil=<seuil>]

Options:  
    -h, --help          show this
    -s, --seuil SEUIL   threshold parameter for nectary detection [default: 13_995]
    --hide              hide the intermediate images (default)
    --show              visualization of intermediate images
    --path PATH         path of the directory containing images to analyse [default: ./datasets/]
    --type TYPE         dictionary of "types" of flowers to process, as writen
                        in the names of images. ex: {'Ma':'Male', 'Fe':'Female',...}
    --quiet             print less text (default)
    --verbose           print more text
"""



import os
import glob
import re
import time
import PIL
import numpy as np
import matplotlib.pyplot as pp
from datetime import date
from docopt import docopt
from scipy import interpolate
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes
from scipy.ndimage import distance_transform_edt, generate_binary_structure, median_filter
from skimage import measure
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from tkinter.filedialog import askdirectory
from tqdm import tqdm



#~~~ Global variables ~~~

#~~ Variables for regular expression for files to process selection ~~
TYPE = ["FFS", "FHS", "FMS"] # respectively for Female, Hermaphrodite and Male Flower
DICOTYPE = {'FFS':'female', 'FHS':'hermaphrodite', 'FMS':'male'}
REGEXP = ".*Tra0000(?P<num>[0-9]+).tif"
REGEXP2 = ".*ini_(?P<num>[0-9]+).tif"
REGEXPFLOWER = 'f".*(?P<flower>{self.curr_type}[0-9][0-9]).*"'

#~~ Image processing ~~
STRUCT8 = generate_binary_structure(2, 2)   # 2D structuring element for mathematical morphology
STRUCT26 = generate_binary_structure(3, 2)  # 3D structuring element for mathematical morphology
SEUIL = 13_995                              # threhold to apply

open3D = 25     # size of the 3D opening
fillholes = False

#~~ Other variables ~~
date = date.today().isoformat() # date of the day used to create a new path
dir_day = f"Results_{date}"     # name of the new path for results



class All_Flowers():
    """ class for the entire datasets. It allows:
        - the creation of a list of all the flowers to process sorted by type (sexe)
        - the creation of a dictionary containing all the infos about all the flowers
        - the processing of each dataset listed"""
    def __init__(self, path2analyze, showimg=False, verb = False, info_file = "infos_flowers.xlsx"):
        self.dico               = {}            # dictionary with infos on each flower to process
        self.path2analyze       = path2analyze  # path to analyse
        self.info_file          = info_file     # Excel file with infos on each flower
        self.list_flowers       = []            # list of the flowers of each type to analyse
        self.curr_type          = TYPE[0]       # current type of flower (Male, Female, Hermaphrodite)
        self.curr_flower        = ""            # name of the current flower
        self.num_flower         = 1             # ID of the current flower
        self.curr_path_flower   = ""            # path of the current flower
        self.showimg            = showimg       # show or hide intermediate images 
        self.dirpp              = False         # path to save intermediate images
        self.create_img_dico()
        self.create_list_flowers()
        self.verb               = verb
    
    def create_img_dico(self):
        """ Creation of the dictionary containing all infos on each flower """
        #~~ Change of path in order to import "Dico_images" ~~
        file = self.info_file
        path_codes = os.path.dirname(os.path.abspath(__file__))
        os.chdir(os.path.dirname(path_codes))
        from Read_flowers_Excel_file import Dico_images
        #~~ Creation of dictionary containing flower infos ~~
        temp = path_codes.split(os.sep)[-1]
        self.dico = Dico_images(f"{path_codes.replace(temp,'data_infos')}{os.sep}{file}")
        
    def create_list_flowers(self):
        """ Creation of the list of "flower" directory to analyse, i.e. all the
        directory in the path """
        self.list_flowers = []
        for self.curr_type in TYPE:
            motif = f"{os.sep}*{self.curr_type}*"
            temp = []
            ind = 4
            while not len(temp):
                if ind == 0:
                    break
                temp = glob.glob(f"{self.path2analyze+motif*ind}")
                ind-=1
            if os.path.isfile(temp[0]):
                temp = glob.glob(f"{self.path2analyze+motif*ind}")
            self.list_flowers.append(temp)
        
    def run_processing(self):
        """ Starts the processing of all the flowers on each list of type """
        for i, self.curr_type in enumerate(TYPE):
            if self.verb: start_time = time.time()
            if not(len(self.list_flowers[i])):
                print(f"No images for flowers of type {self.curr_type}")
                continue
            pbar = tqdm(total = int(len(self.list_flowers[i])),
                        desc = f'Image processing on {DICOTYPE[self.curr_type]} flowers')
            for self.curr_path_flower in self.list_flowers[i]:
                if os.path.isfile(self.curr_path_flower): continue
                """ process each flower of the list for the current type """
                self.curr_flower = Single_Flower(self.curr_type, self.showimg,
                                                 self.curr_path_flower, self.dirpp, self.verb)
                self.curr_flower.info_flower = self.dico[self.curr_flower.flower_name]
                if self.verb:
                    t_begProg = time.time()
                    t = time.localtime(t_begProg)
                    print(f'Current flower : {self.curr_flower.flower_name}')
                    print(f"Progam started at : {t.tm_hour}:{t.tm_min}:{t.tm_sec}"+
                          f" on {t.tm_mon}/{t.tm_mday}/{t.tm_year}")
                self.curr_flower.flower_processing_by_type()
                if self.verb:
                    print(f"End of process for the flower of the directory {self.curr_path_flower}\n")
                    t_finProg = time.time()
                    print("program duration : "+str(t_finProg-t_begProg))
                    t2 = time.localtime(t_finProg)
                    print(f"End of program : {t2.tm_hour}:{t2.tm_min}:{t2.tm_sec}"+
                          f" on {t2.tm_mon}/{t2.tm_mday}/{t2.tm_year}")
                pbar.update()
            if self.verb:
                print(f"Execution duration for {DICOTYPE[self.curr_type]} flowers: "+
                      f"{time.time() - start_time} secondes ---" )
            pbar.close()
        


class Single_Flower():
    """ It is used for each single flower. It allows to:
        - read the ID of the flower
        - create an image of "height" of the flower inside the ROI
        - remove outliers inside the image of "height" to ease the segmentation
        - start the nectary segmentation with different methods depending on\
        the type/sexe of the current flower
        - display of intermediate images
        """
    def __init__(self, type_f, display, curr_path, dirpp=False, dico={}, verbose=False):
        self.curr_type      = type_f
        self.curr_path      = f'{curr_path}{os.sep}'
        self.showimg        = display
        self.dirpp          = dirpp
        self.info_flower    = dico
        self.verb           = verbose
        self.read_flower_name()
        self.list_images()
        if DICOTYPE[self.curr_type] in ['female','hermaphrodite']:
            self.curr_flower = Female_Flower(self.flower_name, self.list_files, self.showimg, self.verb)
        else:
            self.curr_flower = Male_Flower(self.flower_name, self.list_files, self.showimg, self.verb)
        self.curr_flower.dir_res = self.dir_res
        self.blur = self.curr_flower.blur

    def display_image(self, image, label=False, namesave=False):
        if not self.showimg:
            return
        """ Image display with or without the label of each region. 
        The image is saved if 'namesave' is given."""
        pp.imshow(image)
        if label:
            prop = measure.regionprops(image)
            for i, p in enumerate(prop):
                pp.text(int(p.centroid[1]), int(p.centroid[0]), s=str(i+1))
        if namesave:
            pp.savefig(self.dirpp+namesave+'.png')
        pp.show()

    def read_flower_name(self):
        """ extract the name of the flower and its number from the directory
        name """
        path_temp           = os.path.basename(self.curr_path[0:-1])
        regexp               = eval(REGEXPFLOWER)
        matchflower         = re.search(f'{regexp}', path_temp)
        self.flower_name    = matchflower.group('flower')
        self.flower_num     = self.flower_name[len(self.curr_type):]
    
    def list_images(self):
        self.list_files = glob.glob(self.curr_path+'*Tra0000[0-9][0-9][0-9][0-9].tif')
        self.dir_res = f'{self.curr_path}segmentation_nectary{os.sep}{dir_day}{os.sep}'
        if not os.path.exists(self.dir_res):
            os.makedirs(self.dir_res)

    def image_height(self):
        """ Image of height: show the max "height" of each element of the flower """
        self.height = PIL.Image.open(self.list_files[0])
        self.height = np.zeros((self.height.size[1], self.height.size[0]), dtype=np.int16)
        for ifile, file in enumerate(tqdm(self.list_files)):
            match2 = re.search(REGEXP, file)
            if not int(match2.group('num')) in range(self.firstfile, self.lastfile): continue
            img2 = np.array(PIL.Image.open(file))
            img2 = median_filter(img2,size=5)
            img2 = np.where(np.array(img2) >= SEUIL, (ifile+1), 0)
            if int(match2.group('num')) == self.firstfile:
                img2 = binary_fill_holes( np.where(np.array(img2) >= 1, 1, 0))*(ifile+1)
                self.height = img2[:][:]
                continue
            img2b = np.logical_and(np.where(img2==(ifile+1), 1, 0),
                                   np.where(self.height==0, 1, 0))
            tempfich = img2 - self.height
            self.height = np.where(img2b==True, img2, self.height)
            self.height = np.where(tempfich == 1, img2, self.height)
            self.height = np.where(tempfich == 2, img2, self.height)
        self.height_blur = median_filter(self.height, size=self.blur)
    
    def remove_outliers(self, image, radius=2, thresh=50, which='bright'):
        """ Remove outliers (by default bright). For dark outliers: which='dark'.
        Replace the outliers by the median of the neighborhood. """
        med = median_filter(image, size=int(radius))
        difmed = med-image
        if which == 'bright':
            indices = np.where(difmed <= -thresh)
        elif which == 'dark':
            indices = np.where(difmed >= thresh)
        image_b = image[:, :]
        image_b[indices] = med[indices]
        return image_b[:, :]
    
    def flower_processing_by_type(self):
        """ Process the flower with a different method for male and for female
        and hermaphrodite flowers """

            
        if self.verb:
            print(f"Segmentation of {DICOTYPE[self.curr_type]} nectaries.\n\
                  Choosing features are:\n\
                  - type of flowers : {self.curr_type};\n\
                  - results directory: {self.dir_res};\n\
                  - blur: {self.blur};\n\
                  - open3D: {open3D};\n\
                  - holefill in 3D: {fillholes}.")
        
        #~~ Common part of the process ~~
        self.firstfile = int(self.info_flower['z min'])
        self.lastfile  = int(self.info_flower['z max'])
        self.curr_flower.first_file = self.firstfile
        self.curr_flower.last_file = self.lastfile
        #
        # "Height" image
        if self.verb: print('image of height')
        self.image_height()
        #
        # Clean black outliers
        self.height = self.remove_outliers(self.height, 5, 50, which='dark')
        
        # Display images
        PIL.Image.fromarray(self.height).save(self.dir_res + "height.tif")
        self.display_image(self.height, namesave=f'{self.curr_flower}_height')
        self.display_image(self.height_blur, namesave=f'{self.curr_flower}_height_blur{self.blur}')
        #
        #~~ End of common part ~~

       # Mask of the flower
        #TODO : voir les différences entre les 2 lignes (1ère F, 2ème M)
        mask = np.where(self.height > 0, 1, 0)
        self.display_image(mask, namesave=f'{self.curr_flower}_mask1')
        mask = np.where(self.height > np.min(self.height), 1, 0)
        self.display_image(mask, namesave=f'{self.curr_flower}_mask2')
        self.display_image(mask, namesave=f'{self.curr_flower}_mask_flower')
        
        if self.verb: print('Nectary segmentation')
        self.curr_flower.label_segmentation(self.height, self.height_blur, mask)
        self.display_image(self.curr_flower.lbl, label=True, namesave=f'{self.curr_flower}_segmentation')
        
        self.curr_flower.process()



class Female_Flower():
    def __init__(self, flower_ID, listfiles, showimg=False, verbose=False):
        self.flower_ID  = flower_ID
        self.list_files = listfiles
        self.first_file = 0
        self.last_file  = len(self.list_files)
        self.blur       = 50
        self.showimg    = showimg
        self.verb       = verbose

    def morphological_segmentation(self, img_height, mask_height, min_dist=120):
        """ Segmentation using watershed """
        local_maxi = peak_local_max(img_height, min_distance=min_dist, indices=False)
        markers = measure.label(local_maxi)
        ws = watershed(-img_height, markers, mask=mask_height)
        return ws
    
    def label_segmentation(self, img_height, img_height_blur, mask):
        self.lbl_blur = self.morphological_segmentation(img_height_blur, mask, 80)
        self.lbl = self.morphological_segmentation(img_height, mask, 80)
    
    def nectary_detection(self):
        # Flower center on blured 'height' image
        labelc_blur = self.lbl_blur[int(self.lbl_blur.shape[0]/2), int(self.lbl_blur.shape[1]/2)]
        center_blur = np.where(self.lbl_blur == labelc_blur, 1, 0)
        center_blur = binary_fill_holes(center_blur)
        # Flower center on initial 'height' image
        labelc = self.lbl[int(self.lbl.shape[0]/2), int(self.lbl.shape[1]/2)]
        center = np.where(self.lbl == labelc, 1, 0)
        center = binary_fill_holes(center)
        #display_image(center_blur, namesave=f'{self.curr_flower}_center_flower')
        print("Nectary")
        #
        # Detection of nectary area in 'height' image by dilation of the center
        nectary = binary_dilation(center_blur, structure=STRUCT8, iterations=50)
        labeln = {}
        clelabeln = labeln.keys()
        indicesnect = np.where(np.logical_xor(nectary,center_blur) == 1)
        for i in range(len(indicesnect[0])):
            if self.lbl_blur[indicesnect[0][i], indicesnect[1][i]] not in clelabeln:
                labeln[self.lbl_blur[indicesnect[0][i], indicesnect[1][i]]] = 1
        #
        nectary = np.zeros(self.lbl_blur.shape)
        for i in clelabeln:
            nectary = np.logical_or(nectary, np.where(self.lbl_blur == i, 1, 0))
        #display_image(nectary)
        self.nectary = np.where(center==0, binary_fill_holes(nectary), 0)
        #display_image(nectary, namesave=f'{curr_flower}_nectaire')
        del center_blur
        PIL.Image.fromarray(np.array(nectary*255, dtype=np.uint8)).save(\
                           self.dir_res + "nectaire_mask.tif")
        del(self.lbl_blur, self.lbl, labeln, labelc_blur, clelabeln, labelc)

    def remove_outliers_contours(image_ini, img_height, nb_label2):
        """ Supprime les valeurs aberrentes des contours pour qu'elles ne soient
        pas prises en compte dans l'interpolation"""
        image_finale = np.zeros(image_ini.shape)
        lab_a_garder = []
        if nb_label2 > 2:
            dico_lab = {}
            for i2 in range(1, nb_label2+1):
                dico_lab[i2] = len(np.where(image_ini == i2)[0])
            value = list(dico_lab.values())
            value.sort()
            while len(value) > 2:
                value.pop(0)
            for k2, i2 in dico_lab.items():
                if i2 in value:
                    lab_a_garder.append(k2)
        else:
            lab_a_garder = [1, 2]
        for lab in lab_a_garder:
            ind_intervalle = np.where(image_ini == lab)
            meanInt = np.mean(img_height[ind_intervalle])
            stdInt = np.std(img_height[ind_intervalle])
            ind_suppr = np.where(img_height > meanInt+(3*stdInt))
            temp = np.where(image_ini==lab, img_height, 0)
            temp[ind_suppr] = 0
            image_finale = np.maximum(temp, image_finale)
        return image_finale

    def nectary_contours(self):
        print("Contours")
        #
        # Contours de la nectaire afin de réaliser l'interpolation pour le contour inférieur de celle-ci
        nectary_cont = np.logical_xor(self.nectary, binary_erosion(self.nectary, structure=STRUCT8))
        self.ind_nectary = np.where(self.nectary > 0)
        del self.nectary
        label_nect_cont, numlab = measure.label(nectary_cont, return_num=True)
        #display_image(label_nect_cont, namesave=f'{self.curr_flower}_contours_nectaire_lab')
        #
        # Nettoyage des contours par élimination des outliers pour éviter les problèmes d'interpolation
        nectary_cont = self.remove_outliers_contours(label_nect_cont, self.height, numlab)
        #display_image(nectary_cont, namesave=f'{self.curr_flower}_contours_nectaire')
        #
        # Interpolation
        grid_xn, grid_yn = np.indices((nectary_cont.shape[0], nectary_cont.shape[1]))
        points2 = np.where(nectary_cont > 10)
        values2 = nectary_cont[points2]
        points2 = np.array([[points2[0][f], points2[1][f]] for f in range(len(points2[0]))])
        self.interp2 = interpolate.griddata(points2, values2, (grid_xn, grid_yn), \
                                       fill_value=0, method='linear')
        #display_image(interp2, namesave=f'{self.curr_flower}_interpolation')
        del(grid_xn, grid_yn, points2, values2)
        PIL.Image.fromarray(self.interp2).save(self.dir_res + "interpolation.tif")

    def create_3Dimage(self):
        # Création de l'image 3D finale
        imageini = np.zeros([self.nectary_cont.shape[0], self.nectary_cont.shape[1],\
                             int(self.lastfile-self.firstfile)], dtype=np.uint8)
        ind = np.where(self.interp2 > 0)
        print(min(self.interp2[ind]))
        #del(nectaire_cont)
        ind = 0
        for i, file in enumerate(self.list_files):
            match = re.search(REGEXP, file)
            if not int(match.group('num')) in range(self.firstfile, self.lastfile): continue
            img = np.array(PIL.Image.open(file))
            img = np.where(np.array(img) >= SEUIL, 1, 0)
            indtemp = np.where(self.interp2 >= i)
            img[indtemp] = 0
            imageini[:, :, ind] = img[:, :]
            ind += 1
        del(self.interp2, img, indtemp)
        #
        #Changement du 09/09/20 iterations=10 --> iterations = 20
        erosion_bin = binary_erosion(imageini, structure=STRUCT26, iterations=20)
        label_eros, nb_label = measure.label(erosion_bin, return_num=True)
        print(f"nb_label : {nb_label}")
        props = measure.regionprops(label_eros)
        list_lab = []
        for i in props:
            if i["area"] > 500:
                list_lab.append(i["label"])
        print(f"len(list_lab) : {list_lab}")
        for i in range(label_eros.shape[2]):
            if i==0:
                temphist = np.histogram(label_eros[self.ind_nectary[0],self.ind_nectary[1],i], bins=nb_label+1, range=(0, nb_label+1))
            else:
                temphist[0][:] += np.histogram(label_eros[self.ind_nectary[0],self.ind_nectary[1],i], bins=nb_label+1, range=(0, nb_label+1))[0][:]
        a_garder = int(temphist[1][np.where(temphist[0][1:]==temphist[0][1:].max())])+1
        print(temphist)
        print(f"a garder v3 : {a_garder}")

        for i in range(label_eros.shape[2]):
            temp = label_eros[:, :, i]
            temp2 = np.where(temp == a_garder, True, False)
            erosion_bin[:, :, i] = temp2
        print(f"erosion_bin fait")

        label_dil = binary_dilation(erosion_bin, structure=STRUCT26, iterations=20)
        print("label_dil fait")

        #
        del list_lab
        #
        ind = 0
        for i, file in enumerate(self.list_files):
            match = re.search(REGEXP, file)
            if not int(match.group('num')) in range(self.firstfile, self.lastfile): continue
            text = f"nectary_{match.group('num')}.tif"
            ttemp = np.array(np.logical_and(label_dil[:, :, ind],\
                                            imageini[:, :, ind]), dtype=np.uint8)*255
            PIL.Image.fromarray(ttemp).save(self.dir_res + text)
            ind += 1
        #
        #
        del(imageini, file, match, label_dil, text, ttemp)

    def process(self):
        self.nectary_detection()
        self.nectary_contours()
        self.create_3Dimage()



class Male_Flower():
    def __init__(self, flower_ID, listfiles, showimg=False, verbose=False):
        self.flower_ID  = flower_ID
        self.list_files = listfiles
        self.first_file = 0
        self.last_file  = len(self.list_files)
        self.blur       = 15
        self.showimg    = showimg
        self.verb       = verbose
        self.name 		= ""
        
    def morphological_segmentation(self, img_height, mask_height, min_dist=120):
        """ Segmentation using watershed """
        seuilOtsu = threshold_otsu(img_height)
        imgOtsu = np.where(img_height>seuilOtsu,255,0)
        local_maxi = peak_local_max(img_height, min_distance=min_dist, indices=False)
        imgOtsu = np.where(imgOtsu==0,local_maxi,255)
        dist = distance_transform_edt(imgOtsu)
        local_maxi = peak_local_max(dist, min_distance=min_dist, indices=False)
        markers = measure.label(local_maxi)
        local_maxi = watershed(-dist,markers, mask=imgOtsu)
        markers = measure.label(local_maxi)
        ws = watershed(-img_height, markers, mask=mask_height)
        return ws

    def label_segmentation(self, img_height, img_height_blur, mask):
        self.mask = mask
        self.lbl = self.morphological_segmentation(img_height_blur, mask, 80)

    def nectary_detection(self):
        labelc = []
        cont = input("Continue ? (Yes/No/Choose the labels[c]) ")
        if cont in ["Yes","Y", "yes", "y"]:
            del mask
        elif cont in ["Choose", "c", "C", "choose"]:
            labelc = input("Write the labels to keep separated by ',': ")
            labelc = [int(l) for l in labelc.split(',')]
        elif cont in ["No", "no", "N", "n"]:
            labels_ws = self.morphological_segmentation(self.lbl, self.mask, 120)
            #affiche_image(labels_ws, label=True, namesave=f'{curr_fleur}_segmentation_120')
            cont = input("Continuer ? (Oui/Choisir les labels[c]) ")
            if cont in ["Oui","O", "oui", "o"]:
                del self.mask
            elif cont in ["Choisir", "c", "C", "choisir"]:
                labelc = input("Inscrire les labels à conserver séparés par des ',' : ")
                labelc = [int(l) for l in labelc.split(',')]
        if not len(labelc):
            labelc = [labels_ws[int(labels_ws.shape[0]/2), int(labels_ws.shape[1]/2)],]
        nectary = np.zeros(labels_ws.shape)
        for l in labelc:
            nectary = np.logical_or(nectary, np.where(labels_ws == l, 1, 0))
        #affiche_image(nectary)
        self.nectary = binary_fill_holes(nectary)
        #PIL.Image.fromarray(np.array(nectary*255, dtype=np.uint8)).save(\
        #                   dossier_res + f"nectaire_mask_{SEUIL}_blur{blur}.tif")
        #affiche_image(nectary, namesave=f'{curr_fleur}_nectaire')
        
    def remove_outliers_contours(image_ini, img_height, nb_label2):
        """ Supprime les valeurs aberrentes des contours pour qu'elles ne soient
        pas prises en compte dans l'interpolation"""
        image_finale = np.zeros(image_ini.shape)
        lab_a_garder = []
        if nb_label2 > 1:
            dico_lab = {}
            for i2 in range(1, nb_label2+1):
                dico_lab[i2] = len(np.where(image_ini == i2)[0])
            value = list(dico_lab.values())
            value.sort()
            while len(value) > 2:
                value.pop(0)
            for k2, i2 in dico_lab.items():
                if i2 in value:
                    lab_a_garder.append(k2)
        else:
            lab_a_garder = [1, ]
        for lab in lab_a_garder:
            ind_intervalle = np.where(image_ini == lab)
            meanInt = np.median(img_height[ind_intervalle])
            stdInt = np.std(img_height[ind_intervalle])
            ind_suppr = np.where(img_height-meanInt >= (2*stdInt))
            temp = np.where(image_ini==lab, img_height, 0)
            temp[ind_suppr] = 0
            image_finale = np.maximum(temp, image_finale)
        return image_finale

    def nectary_contours(self):
        print("Contours")
        #
        # Contours de la nectary afin de réaliser l'interpolation pour le contour inférieur de celle-ci
        nectary_cont = np.logical_xor(self.nectary, binary_erosion(self.nectary, structure=STRUCT8))
        self.ind_nectary = np.where(self.nectary > 0)
        del self.nectary
        label_nect_cont, numlab = measure.label(nectary_cont, return_num=True)
        #display_image(label_nect_cont, namesave=f'{self.curr_flower}_contours_nectaire_lab')
        #
        # Nettoyage des contours par élimination des outliers pour éviter les problèmes d'interpolation
        nectary_cont = self.remove_outliers_contours(label_nect_cont, self.height, numlab)
        #PIL.Image.fromarray(nectaire_cont).save(\
        #                   dossier_res + f"nectaire_cont_{SEUIL}_blur{blur}.tif")
        #display_image(nectary_cont, namesave=f'{self.curr_flower}_contours_nectaire')
        #
        # Interpolation
        grid_xn, grid_yn = np.indices((nectary_cont.shape[0], nectary_cont.shape[1]))
        points2 = np.where(nectary_cont > 10)
        values2 = nectary_cont[points2]
        points2 = np.array([[points2[0][f], points2[1][f]] for f in range(len(points2[0]))])
        self.interp2 = interpolate.griddata(points2, values2, (grid_xn, grid_yn), \
                                       fill_value=0, method='linear')
        #display_image(interp2, namesave=f'{self.curr_flower}_interpolation')
        del(grid_xn, grid_yn, points2, values2)
        PIL.Image.fromarray(self.interp2).save(self.dir_res + "interpolation.tif")
        
    def create_3Dimage(self):
        # Création de l'image 3D finale
        imageini = np.zeros([self.nectary_cont.shape[0], self.nectary_cont.shape[1],\
                             int(self.lastfile-self.firstfile)], dtype=np.uint8)
        ind = np.where(self.interp2 > 0)
        print(min(self.interp2[ind]))
        ind = 0
        for i, file in enumerate(self.list_files):
            match = re.search(REGEXP, file)
            if not int(match.group('num')) in range(self.firstfile, self.lastfile): continue
            img = np.array(PIL.Image.open(file))
            img = np.where(np.array(img) >= SEUIL, 1, 0)
            indtemp = np.where(self.interp2 >= i)
            img[indtemp] = 0
            imageini[:, :, ind] = img[:, :]
            ind += 1
        del(self.interp2, img, indtemp)
        #affiche_image(imageini[700,:,:])
        #
        if fillholes:
            imageini = binary_fill_holes(imageini, structure=STRUCT26)
        erosion_bin = binary_erosion(imageini, structure=STRUCT26, iterations=open3D)
        #affiche_image(erosion_bin[700,:,:])
        label_eros, nb_label = measure.label(erosion_bin, return_num=True)
        #affiche_image(label_eros[700,:,:])
        print(f"nb_label : {nb_label}")
        histoini = np.histogram(label_eros, bins=nb_label+1, range=(0, nb_label+1))
        for i in range(label_eros.shape[2]):
            if i==0:
                temphist = np.histogram(label_eros[self.ind_nectary[0],self.ind_nectary[1],i], bins=nb_label+1, range=(0, nb_label+1))
            else:
                temphist[0][:] += np.histogram(label_eros[self.ind_nectary[0],self.ind_nectary[1],i], bins=nb_label+1, range=(0, nb_label+1))[0][:]
        histopc = [nect/tot for nect, tot in zip(temphist[0][1:], histoini[0][1:])]
        print(histopc)
        # Modification du 18_01_21 pour diminution progressive du % de présence du label dans l'objet
        pc = 0.75
        a_garder = [int(temphist[1][i+1]) for i in range(len(histopc)) if histopc[i] > pc]
        print(temphist)
        while not(len(a_garder)):
            pc -= 0.05
            print(f"pc : {pc}")
            a_garder = [int(temphist[1][i+1]) for i in range(len(histopc)) if histopc[i] > pc]
        print(f"a garder v3 : {a_garder}")



        for i in range(label_eros.shape[2]):
            temp = label_eros[:, :, i]
            temp2 = np.zeros(temp.shape, dtype=bool)
            for g in a_garder:
                temp2 = np.where(temp==g, True, temp2)
            erosion_bin[:, :, i] = temp2
        print(f"erosion_bin fait")

        label_dil = binary_dilation(erosion_bin, structure=STRUCT26, iterations=open3D+5, mask = imageini)
        #affiche_image(label_dil[700,:,:])
        print("label_dil fait")
        
        #
        #del list_lab
        #
        ind = 0
        for i, file in enumerate(self.list_files):
            match = re.search(REGEXP, file)
            if not int(match.group('num')) in range(self.firstfile, self.lastfile): continue
            if fillholes:
                text = f"nectary_fillholes_{match.group('num')}.tif"
            else:
                text = f"nectary_{match.group('num')}.tif"
            ttemp = np.array(np.logical_and(label_dil[:, :, ind],\
                                            imageini[:, :, ind]), dtype=np.uint8)*255
            PIL.Image.fromarray(ttemp).save(self.dir_res + text)
            ind += 1
        #
        #
        #del(imageini, file, match, label_dil, text, ttemp)
        #del(interp2, img, indtemp)
        #del(fich, match, text, ttemp)

    def process(self):
        self.nectary_detection()
        self.nectary_contours()
        self.create_3Dimage()



def main(argv):
    """Main function"""
    # Initialisation des variables
    global SEUIL

    if "show" in argv.keys():
        showimg = True
    else:
        showimg = False
    if "path" in argv.keys():
        d = argv["path"]
    else:
        curr_path = os.getcwd()
        d = askdirectory(initialdir=curr_path, title='Select the images path')
    if "seuil" in argv.keys():
        SEUIL = argv["seuil"]

    if "verbose" in argv.keys():
        verb = True
    else:
        verb = False
    
    all_flowers = All_Flowers(d, showimg, verb)
    if showimg:
        all_flowers.dirpp = askdirectory(initialdir=d,
                                         title='Select a path to save intermediate images')
        if not os.path.exists(all_flowers.dirpp):
            os.makedirs(all_flowers.dirpp)
    all_flowers.run_processing()
    return all_flowers

    
if __name__ == "__main__":
    # récupèration des options ajoutées en ligne de commande
    arguments = docopt(__doc__)
    # appel de la fonction main
    main(arguments)    

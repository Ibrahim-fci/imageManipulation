import random
from PIL import ImageTk, Image
from scipy import ndimage
from matplotlib import pyplot as plt
import cv2
import tkinter as tk
from tkinter import filedialog
import os
import ctypes
import numpy as np
from PIL import Image, ImageFilter

#This creates the main window of an application
window = tk.Tk()
window.title("FCI Image Manipulation")
window.geometry("1200x900")
window.configure(background='grey')
window.resizable(0,0)


#................global variables with intial values..................
file_path=''
path = "E:\\faculty_fourthYear\\Python_project\\DigitalIMage\\ll.png"
path2 = "E:\\faculty_fourthYear\\Python_project\\DigitalIMage\\ll.png"
im = Image.open(path)
imr=Image.open(path)
img = ImageTk.PhotoImage(im)
imgr=ImageTk.PhotoImage(imr)
dir_path = os.getcwd()
panel = tk.Label(window,bg='gray', text='Image Here',font=('bold',16))
panel.place(x=50, y=70, width=650, height=450)
image_median = cv2.imread(file_path)
roberts_cross_v = np.array( [[ 0, 0, 0 ],
                             [ 0, 1, 0 ],
                             [ 0, 0,-1 ]] )

roberts_cross_h = np.array( [[ 0, 0, 0 ],
                             [ 0, 0, 1 ],
                             [ 0,-1, 0 ]] )



#...........Action Functions................................


def hist():
    global path, im
    try:
        im.save(dir_path + '\\ooppl23.png')
        path = dir_path + '\\ooppl23.png'
        im2 = cv2.imread(path)
        color = ('b', 'g', 'r')
        plt.figure()
        for i, col in enumerate(color):
            histr = cv2.calcHist([im2], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.show()
    except:
        return



def sobel():
    global path, im, img, file_path, panel2, panel, dir_path, image_median, roberts_cross_v, roberts_cross_h
    try:
        im.save(dir_path + '\\ooppl23.png')
        path = dir_path + '\\ooppl23.png'
        im = cv2.imread(path)

        sobely = cv2.Sobel(im,-1,dx=0,dy=1,ksize=5,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)

        cv2.imwrite(dir_path + '\\Mfi1lter.png', sobely)

        path = dir_path + '\\Mfi1lter.png'

        im = Image.open(path)
        v = im.resize((650, 510), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(v)

        panel = tk.Label(window, image=img)
        panel.cget("image") == img
        panel.place(x=50, y=70, width=650, height=510)
    except:
        return


def sobelx():
    global path, im, img, file_path, panel2, panel, dir_path, image_median, roberts_cross_v, roberts_cross_h
    try:
        im.save(dir_path + '\\ooppl23.png')
        path = dir_path + '\\ooppl23.png'
        im = cv2.imread(path)

        sobely = cv2.Sobel(im,-1,dx=1,dy=0,ksize=5,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)

        cv2.imwrite(dir_path + '\\Mfi1lter.png', sobely)

        path = dir_path + '\\Mfi1lter.png'

        im = Image.open(path)
        v = im.resize((650, 510), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(v)

        panel = tk.Label(window, image=img)
        panel.cget("image") == img
        panel.place(x=50, y=70, width=650, height=510)
    except:
        return






def laplacian():
    global path, im, img, file_path, panel2, panel, dir_path, image_median, roberts_cross_v, roberts_cross_h
    try:
        im.save(dir_path + '\\ooppl23.png')
        path = dir_path + '\\ooppl23.png'
        im = cv2.imread(path)

        sobely = cv2.Laplacian(im,-1,ksize=7,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)

        cv2.imwrite(dir_path + '\\Mfi1lter.png', sobely)

        path = dir_path + '\\Mfi1lter.png'

        im = Image.open(path)
        v = im.resize((650, 510), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(v)

        panel = tk.Label(window, image=img)
        panel.cget("image") == img
        panel.place(x=50, y=70, width=650, height=510)
    except:
        return










def save_image( data, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(data,0,255), dtype="uint8"), "L" )
    img.save( outfilename )

def roberts_cross() :
     global path, im, img, file_path, panel2, panel, dir_path, image_median,roberts_cross_v,roberts_cross_h
     try:
         path = file_path

         im = cv2.imread(path, 0)

         vertical = ndimage.convolve(im, roberts_cross_v)
         horizontal = ndimage.convolve(im, roberts_cross_h)

         output_image = np.sqrt( np.square(horizontal) + np.square(vertical))
         save_image(output_image, dir_path + '\\roper.png')

         path = dir_path + '\\roper.png'

         im = Image.open(path)
         v = im.resize((650, 510), Image.ANTIALIAS)
         img = ImageTk.PhotoImage(v)

         panel = tk.Label(window, image=img)
         panel.cget("image") == img
         panel.place(x=50, y=70, width=650, height=510)
     except:
         ctypes.windll.user32.MessageBoxW(0, "There Is No Image To Save You Must Open Image First...",
                                          "Saving Confirmation", 1)
         return




def edgeEnhanceMore():
    global path, im, img, file_path, panel2, panel, dir_path
    try:
            if file_path =='':
                path = file_path
                im = Image.open(path)


            im = im.filter(ImageFilter.EDGE_ENHANCE_MORE)
            im.save(dir_path+'\\ooppl23.png')
            path = dir_path + '\\ooppl23.png'

            im = Image.open(path)
            v = im.resize((650, 510), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(v)

            panel = tk.Label(window, image=img)
            panel.cget("image") == img
            panel.place(x=50, y=70, width=650, height=510)

    except:
        ctypes.windll.user32.MessageBoxW(0, "There Is No Image To Save You Must Open Image First..." , "Saving Confirmation", 1)
        return




def edgeEnhance():
    global path, im, img, file_path, panel2, panel, dir_path
    try:
            if file_path == '':
                path = file_path
                im = Image.open(path)


            im = im.filter(ImageFilter.EDGE_ENHANCE)
            im.save(dir_path+'\\ooppl23.png')
            path = dir_path + '\\ooppl23.png'

            im = Image.open(path)
            v = im.resize((650, 510), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(v)

            panel = tk.Label(window, image=img)
            panel.cget("image") == img
            panel.place(x=50, y=70, width=650, height=510)

    except:
        ctypes.windll.user32.MessageBoxW(0, "There Is No Image To Save You Must Open Image First..." , "Saving Confirmation", 1)
        return




def BlureFilter():
    global path, im, img, file_path, panel2, panel, dir_path
    try:

            if file_path == '':
                path = file_path
                im = Image.open(path)


            im = im.filter(ImageFilter.BLUR)
            im.save(dir_path+'\\ooppl23.png')
            path = dir_path + '\\ooppl23.png'

            im = Image.open(path)
            v = im.resize((650, 510), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(v)

            panel = tk.Label(window, image=img)
            panel.cget("image") == img
            panel.place(x=50, y=70, width=650, height=510)

    except:
        ctypes.windll.user32.MessageBoxW(0, "There Is No Image To Save You Must Open Image First..." , "Saving Confirmation", 1)
        return




def grayScaleFilter():
    global path, im, img, file_path, panel2, panel, dir_path
    try:
            if file_path == '':
                path = file_path
                im = Image.open(path)


            im=im.convert('LA')
            im.save(dir_path+'\\ooppl23.png')
            path = dir_path + '\\ooppl23.png'
            im = Image.open(path)
            v = im.resize((650, 510), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(v)

            panel = tk.Label(window, image=img)
            panel.cget("image") == img
            panel.place(x=50, y=70, width=650, height=510)

    except:
        ctypes.windll.user32.MessageBoxW(0, "There Is No Image To Save You Must Open Image First...", "Saving Confirmation", 1)
        return



def sp_noise():
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''

    global path, im, img, file_path, panel2, panel, dir_path, image_median
    try:

        im.save(dir_path + '\\noiiiisy.png')
        path = dir_path + '\\noiiiisy.png'
        image = cv2.imread(path, 0)

        prob=0.05
        output = np.zeros(image.shape,np.uint8)
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        cv2.imwrite(dir_path + '\\Noisy.jpg', output)
        path = dir_path + '\\Noisy.jpg'
        im = Image.open(path)
        v = im.resize((650, 510), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(v)
        panel = tk.Label(window, image=img)
        panel.cget("image") == img
        panel.place(x=50, y=70, width=650, height=510)
        return
    except:
        ctypes.windll.user32.MessageBoxW(0, "There Is No Image To Save You Must Open Image First...",
                                         "Saving Confirmation", 1)
        return





def none():
    global path, im, img, file_path, panel2, panel, dir_path, image_median
    try:
        path = file_path
        im = Image.open(path)
        v = im.resize((650, 510), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(v)
        panel = tk.Label(window, image=img)
        panel.cget("image") == img
        panel.place(x=50, y=70, width=650, height=510)
    except:
        ctypes.windll.user32.MessageBoxW(0, "You Must choose an image", "Saving Confirmation", 1)
        return




def gussein():
    global path, im, img, file_path, panel2, panel, dir_path, image_median
    try:
        im.save(dir_path + '\\gggg.png')
        path = dir_path + '\\gggg.png'
        image_G = cv2.imread(path)
        # apply the 3x3 median filter on the image
        processed_image = cv2.GaussianBlur( image_G,(5,5),0)
        cv2.imwrite(dir_path + '\\Mfilter.png', processed_image)

        path = dir_path + '\\Mfilter.png'

        im = Image.open(path)
        v = im.resize((650, 510), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(v)

        panel = tk.Label(window, image=img)
        panel.cget("image") == img
        panel.place(x=50, y=70, width=650, height=510)
    except:
        ctypes.windll.user32.MessageBoxW(0, "There Is No Image To Save You Must Open Image First...",
                                         "Saving Confirmation", 1)
        return





def laplace_of_gaussian(gray_img, sigma=1., kappa=0.75, pad=False):
   try:
        assert len(gray_img.shape) == 2
        img = cv2.GaussianBlur(gray_img, (0, 0), sigma) if 0. < sigma else gray_img
        img = cv2.Laplacian(img, cv2.CV_64F)
        rows, cols = img.shape[:2]
        # min/max of 3x3-neighbourhoods
        min_map = np.minimum.reduce(list(img[r:rows-2+r, c:cols-2+c]
                                         for r in range(3) for c in range(3)))
        max_map = np.maximum.reduce(list(img[r:rows-2+r, c:cols-2+c]
                                         for r in range(3) for c in range(3)))
        # bool matrix for image value positiv (w/out border pixels)
        pos_img = 0 < img[1:rows-1, 1:cols-1]
        # bool matrix for min < 0 and 0 < image pixel
        neg_min = min_map < 0
        neg_min[1 - pos_img] = 0
        # bool matrix for 0 < max and image pixel < 0
        pos_max = 0 < max_map
        pos_max[pos_img] = 0
        # sign change at pixel?
        zero_cross = neg_min + pos_max
        # values: max - min, scaled to 0--255; set to 0 for no sign change
        value_scale = 255. / max(1., img.max() - img.min())
        values = value_scale * (max_map - min_map)
        values[1 - zero_cross] = 0.
        # optional thresholding
        if 0. <= kappa:
            thresh = float(np.absolute(img).mean()) * kappa
            values[values < thresh] = 0.
        log_img = values.astype(np.uint8)
        if pad:
            log_img = np.pad(log_img, pad_width=1, mode='constant', constant_values=0)
        return log_img
   except:
       ctypes.windll.user32.MessageBoxW(0, "There Is No Image To Save You Must Open Image First...",
                                        "Saving Confirmation", 1)
       return



def _main():
    global path, im, img, file_path, panel2, panel, dir_path, image_median
    try:
        im.save(dir_path + '\\ll.png')
        path = dir_path + '\\ll.png'
        img = cv2.imread(path, 1)
        """Test routine"""
        # load grayscale image
        # lena removed from newer scipy versions
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # apply LoG
        log = laplace_of_gaussian(img)
        cv2.imwrite(dir_path + '\\Lablacian.png', log)

        path = dir_path + '\\Lablacian.png'
        im = Image.open(path)
        v = im.resize((650, 510), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(v)

        panel = tk.Label(window, image=img)
        panel.cget("image") == img
        panel.place(x=50, y=70, width=650, height=510)
    except:
        ctypes.windll.user32.MessageBoxW(0, "There Is No Image To Save You Must Open Image First...",
                                         "Saving Confirmation", 1)
        return




def cannyEdgeDetection():
    global path, im, img, file_path, panel2, panel, dir_path, image_median
    try:
        im.save(dir_path + '\\ll.png')
        path = dir_path + '\\ll.png'
        image_median = cv2.imread(path)
        # apply the 3x3 median filter on the image
        edges = cv2.Canny(image_median, 100, 200)
        cv2.imwrite(dir_path + '\\Canny.png', edges)

        path = dir_path + '\\Canny.png'

        im = Image.open(path)
        v = im.resize((650, 510), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(v)

        panel = tk.Label(window, image=img)
        panel.cget("image") == img
        panel.place(x=50, y=70, width=650, height=510)
    except:
        ctypes.windll.user32.MessageBoxW(0, "There Is No Image To Save You Must Open Image First...", "Saving Confirmation", 1)
        return





def medianfilter():
    global path, im, img, file_path, panel2, panel, dir_path,image_median
    try:
        im.save(dir_path + '\\ll.png')
        path = dir_path + '\\ll.png'
        image_median = cv2.imread(path)
        # apply the 3x3 median filter on the image
        processed_image = cv2.medianBlur(image_median, 3)
        cv2.imwrite(dir_path + '\\Mfilter.png', processed_image)

        path = dir_path + '\\Mfilter.png'

        im = Image.open(path)
        v = im.resize((650, 510), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(v)

        panel = tk.Label(window, image=img)
        panel.cget("image") == img
        panel.place(x=50, y=70, width=650, height=510)
    except:
        ctypes.windll.user32.MessageBoxW(0, "There Is No Image To Save You Must Open Image First...","Saving Confirmation", 1)
        return





def meanFilter():
    global path, im, img, file_path, panel2, panel, dir_path,image_median
    try:
        im.save(dir_path + '\\ll.png')
        path = dir_path + '\\ll.png'
        image_mean = cv2.imread(path)
        # apply the 3x3 mean filter on the image
        kernel = np.ones((3, 3), np.float32) / 9
        processed_image = cv2.filter2D(image_mean, -1, kernel)
        cv2.imwrite(dir_path + '\\Mfilter.png', processed_image)

        path = dir_path + '\\Mfilter.png'

        im = Image.open(path)
        v = im.resize((650, 510), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(v)

        panel = tk.Label(window, image=img)
        panel.cget("image") == img
        panel.place(x=50, y=70, width=650, height=510)
    except:
        ctypes.windll.user32.MessageBoxW(0, "There Is No Image To Save You Must Open Image First...","Saving Confirmation", 1)
        return




def file_path_Save():
        try:
            if path != "E:\\faculty_fourthYear\\Python_project\\DigitalIMage\\ll.png":
                file = filedialog.asksaveasfile(mode='w', defaultextension=".png")
            else:
                ctypes.windll.user32.MessageBoxW(0, "There Is No Image To Save You Must Open Image First...", "Saving Confirmation", 1)
                return

            try:
                im.save(file.name)
                ctypes.windll.user32.MessageBoxW(0, "Image Saved successfully", "Saving Confirmation", 1)
            except:
                ctypes.windll.user32.MessageBoxW(0, "You Must choose an image Name", "Saving Confirmation", 1)
                return
        except:
            ctypes.windll.user32.MessageBoxW(0, "There Is No Image To Save", "Saving Confirmation", 1)
            return





def OpenImage():
    global path, im, img, file_path, original,imgr,imr
    file_path = filedialog.askopenfilename()
    try:
        path = file_path
        im = Image.open(path)
        imr=im
        v = im.resize((650, 510), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(v)
        panel = tk.Label(window, image=img)
        panel.cget("image") == img
        panel.place(x=50, y=70, width=650, height=510)

        vr = imr.resize((410, 250), Image.ANTIALIAS)
        imgr = ImageTk.PhotoImage(vr)
        original = tk.Label(window, bg='gray', text='Original Image',image=imgr)
        original.cget("image") == imgr
        original.place(x=770, y=630, width=410, height=250)
    except:
        ctypes.windll.user32.MessageBoxW(0, "You Must choose an image", "Saving Confirmation", 1)
        return






#............Action Buttons............................................

nav = tk.Label(window,bg='black',fg='white',text='Fci Team Image processing System', font=('bold',16))
nav.place(x=0, y=0, width=1200, height=40)

w = tk.Button( window,command=cannyEdgeDetection, bg='blue',fg='white',justify='left',text=' Canny Edge Detector')
w.place( x=980,y=70,width=200, height=60)



w = tk.Button( window,command=_main, bg='blue',fg='white',justify='left',text='laplacian Of Gaussian EdgeDetector',)
w.place( x=980,y=150,width=200, height=60)


w = tk.Button( window,command=grayScaleFilter, bg='blue',fg='white',justify='left',text='GrayScale Filter')
w.place( x=770,y=310,width=200, height=60)


w2 = tk.Button( window,command=medianfilter, bg='blue',fg='white',justify='left',text='Median Filter')
w2.place( x=770,y=390,width=200, height=60)


w = tk.Button( window,command=BlureFilter, bg='blue',fg='white',justify='left',text='BLUR')
w.place( x=770,y=70,width=200, height=60)


w = tk.Button( window,command=edgeEnhance, bg='blue',fg='white',justify='left',text='EdgeEnhancement')
w.place( x=980,y=230,width=200, height=60)


w = tk.Button( window,command=edgeEnhanceMore, bg='blue',fg='white',justify='left',text='EdgeEnhance More')
w.place( x=980,y=310,width=200, height=60)


w = tk.Button( window,command=gussein, bg='blue',fg='white',justify='left',text=' GaussianBlur ')
w.place( x=770,y=150,width=200, height=60)


w = tk.Button( window,command=meanFilter, bg='blue',fg='white',justify='left',text='Average Filter ')
w.place( x=770,y=230,width=200, height=60)





w = tk.Button( window,command=sp_noise, bg='blue',fg='white',justify='left',text='Salt&Pepper Noise')
w.place( x=770,y=470,width=200, height=60)

w2 = tk.Button( window,command=sobel, bg='blue',fg='white',justify='left',text='Sobely edge Detector')
w2.place( x=980,y=550,width=200, height=60)

w2 = tk.Button(window,command=laplacian, bg='blue',fg='white',justify='left',text='Lablacian')
w2.place( x=980,y=390,width=200, height=60)


w2 = tk.Button(window,command=roberts_cross, bg='blue',fg='white',justify='left',text='Roberts Edge Detector')
w2.place( x=980,y=470,width=200, height=60)



w2 = tk.Button(window,command=sobelx, bg='blue',fg='white',justify='left',text='Sobelx edge Detector')
w2.place( x=770,y=550,width=200, height=60)






original = tk.Label(window,bg='gray',text='Original Image', font=('bold',16))
original.place(x=770, y=630, width=410, height=250)


w = tk.Button(window,command=OpenImage, bg='green',fg='white',text='Open Image',font=('bold',14))
w.place( x=50,y=610,width=650, height=60)


w = tk.Button(window, command=file_path_Save,bg='green',fg='white',text='Save Image',font=('bold',14))
w.place( x=50,y=680,width=650, height=60)


w = tk.Button(window, command=hist,bg='dark goldenrod',fg='white',text='Color Histogram',font=('bold',14))
w.place( x=50,y=750,width=650, height=60)

w = tk.Button(window, command=none,bg='red',fg='white',text='None',font=('bold',14))
w.place( x=50,y=820,width=650, height=60)

#Start the GUI

window.mainloop()

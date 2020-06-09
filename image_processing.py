import numpy as np
import cv2
from PIL import Image
import os

#helper functions
def create_image(i, j):
  image = Image.new("RGB", (i, j), "white")
  return image

def open_image(path):
  newImage = Image.open(path)
  return newImage

def PolyArea2D(pts): #http://stackoverflow.com/questions/19873596/convex-hull-area-in-python
    lines = np.hstack([pts,np.roll(pts,-1,axis=0)])
    area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
    return area

#1 - convert to grayscale
def convert_grayscale(image):
  # Get size
  width, height = image.size

  # Create new Image and a Pixel Map
  new = create_image(width, height)
  pixels = new.load()
  pix_val_R = list(image.getdata(0))
  pix_val_G = list(image.getdata(1))
  pix_val_B = list(image.getdata(2))
  no_pix = len(pix_val_R)
  grays = []

  # get all grays
  for i in range(no_pix):
      #get colors
      red = pix_val_R[i]
      green = pix_val_G[i]
      blue = pix_val_B[i]

      #transform to grayscale
      gray = (red * 0.299) + (green * 0.587) + (blue * 0.114)
      gray = (red+green+blue) / 3
      grays.append(int(gray))

  #change image to gray
  for i in range(height):
      for j in range(width):
          it = i*width + j
          gray = grays[it]
          pixels[j, i] = (int(gray), int(gray), int(gray))
  return new

#2 - find border
def find_border(path):
    orig = cv2.imread(path)
    o2 = orig.copy()
    img = cv2.imread(path,0)
    ret3,th3 = cv2.threshold(img,250,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    im2, contours, hierarchy = cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    alist = []
    biggestcontour = []
    for contour in contours:
        area=PolyArea2D(contour[:,0])
        alist.append(area)
        mx=max(alist) #largest contour
        if mx==area:
            biggestcontour=contour

    #finds bounding box of contour
    rect = cv2.minAreaRect(np.array(biggestcontour))
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(o2,[box],0,(0,0,255),5)

    return o2

#3 - crop
def crop_image(image):
    width, height = image.size

    pix_val_R = list(image.getdata(0))
    pix_val_G = list(image.getdata(1))
    pix_val_B = list(image.getdata(2))

    #LEFT
    left_height = height/2
    pos_from_left = 0
    red_found = 0
    l = 0;

    while red_found == 0:
        pos = int((left_height*width) + pos_from_left)
        if pix_val_R[pos] == 255 and pix_val_G[pos] == 0 and pix_val_B[pos] == 0:
            red_found = 1
            l = pos_from_left%width
            break
        pos_from_left = pos_from_left + 1

    #UPPER
    up_width = width/2
    pos_from_up = 0
    red_found = 0
    u = 0

    while red_found == 0:
        pos = int(pos_from_up*width + up_width)
        if pix_val_R[pos] == 255 and pix_val_G[pos] == 0 and pix_val_B[pos] == 0:
            red_found = 1
            u = pos_from_up
            break
        pos_from_up = pos_from_up + 1

    #RIGHT
    right_height = height/2
    pos_from_right = width
    red_found = 0
    r = 0

    while red_found == 0:
        pos = int((right_height*width) + pos_from_right)
        if pix_val_R[pos] == 255 and pix_val_G[pos] == 0 and pix_val_B[pos] == 0:
            red_found = 1
            r = pos_from_right%width
            break
        pos_from_right = pos_from_right - 1

    #DOWN
    down_width = width/2
    pos_from_down = height-1
    red_found = 0
    d = 0

    while red_found == 0:
        pos = int(pos_from_down*width + down_width)
        if pix_val_R[pos] == 255 and pix_val_G[pos] == 0 and pix_val_B[pos] == 0:
            red_found = 1
            d = pos_from_down
            break
        pos_from_down = pos_from_down - 1

    #PERFORM CROP
    return image.crop((l+10, u+10, r-10, d-10))

#4 - convert to black and white
def convert_bw(image):
    # Get size
    width, height = image.size

    # Create new Image and a Pixel Map
    img2 = create_image(width, height)
    pixels = img2.load()
    pix_val_gray = list(image.getdata(0))
    no_pix = len(pix_val_gray)
    bws = []

    # get all grays
    for i in range(no_pix):
        #get gray val
        gray = pix_val_gray[i]

        #transform to b/w
        bw = 0
        if gray > 100:
            bw = 255
        else:
            bw = 0

        bws.append(int(bw))

    #change image to bw
    for i in range(height):
        for j in range(width):
            it = i*width + j
            bw = bws[it]
            pixels[j, i] = (int(bw), int(bw), int(bw))
    return img2

#5 - quantify markings
def quantify(img):
    width, height = img.size
    marks = 0

    pix_val_R = list(img.getdata(0))
    pix_val_G = list(img.getdata(1))
    pix_val_B = list(img.getdata(2))

    for i in range(height):
        for j in range(width):
            pos = (i*width) + j
            R = pix_val_R[pos]
            G = pix_val_G[pos]
            B = pix_val_B[pos]

            if R==0 and G==0 and B==0:
                marks = marks+1

    return marks/(width*height)

def last_crop_image(image):
    width, height = image.size

    pix_val_R = list(image.getdata(0))
    pix_val_G = list(image.getdata(1))
    pix_val_B = list(image.getdata(2))

    ideal_ratio = 2

    if (height*ideal_ratio > width):
        w = width
        l = w/2
    else:
        l = height
        w = l*2

    return image.crop((0, 0, w, l))

#main
def main():
    #path = "Picture1.png"
    #output_path = "Picture2.png"
    #img = open_image(path)
    #img_gray = convert_grayscale(img)
    #img_gray.save("temp.png", 'png')
    ##img_gray.save("gray.png", 'png')
    #img_border = find_border("temp.png")
    #cv2.imwrite("temp2.png", img_border)
    ##cv2.imwrite("border.png", img_border)
    #os.remove("temp.png")
    #img_border = open_image("temp2.png")
    #img_cropped = crop_image(img_border)
    ##img_cropped.save("cropped.png", 'png')
    #os.remove("temp2.png")
    #img_bw = convert_bw(img_cropped)
    #img_bw.save(output_path, 'png')
    #q = quantify(img_bw)
    #print("% markings on image = " + str(q))

    input_path = "Picture1.png"
    output_path = "Picture2.png"
    img2 = open_image(input_path)
    img_gray = convert_grayscale(img2)
    img_bw = convert_bw(img_gray)
    b = last_crop_image(img_bw)
    width, height = b.size

    #resize image
    basewidth = 500
    wpercent = (basewidth/float(b.size[0]))
    hsize = int((float(b.size[1])*float(wpercent)))
    b2 = b.resize((basewidth,hsize), Image.ANTIALIAS)
    b2.save(output_path, 'png')

if __name__ == "__main__":
    main()

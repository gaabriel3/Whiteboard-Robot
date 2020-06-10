# Whiteboard-Robot

**Step 1:**  
a) **Within image_processing.py**, if you have a whiteboard created digitally, without the need for processing a photo, please change:  
LINE 239 -> Input the source of your whiteboard image  
LINE 240 -> Input the path desired for the outputted image  

b) **Within image_processing.py**, if you have a photo of the whiteboard, with the need for processing this photo, please change:  
LINES 220-237 -> uncomment all lines with only one '#'  
LINES 239-243 -> comment out  
LINE 220 -> Input the source of your whiteboard image  
LINE 221 -> Input the path desired for the outputted image  

**Step 2:**  
Within **path_finder_updated.py**, please change:  
LINE 875 -> comment out  
LINE 876 -> Input the path for the image resulting in Step 1  
When executing this file, output the file to a text file.  
A file such as the output.txt files in the example folders will be the result.  

**Step 3:**  
Within **sim.py**, for your use, please change:   
LINE 139 -> Input the source of your whiteboard image  
LINE 475 -> Input the cartesian path plan coordinates generated in Step 2   
LINE 497 -> Input the source of your whiteboard image   

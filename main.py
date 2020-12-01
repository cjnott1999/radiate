from PIL import Image
from tomographic_functions import *
from display_functions import *


image = Image.open('Final Project/cat.jpg')
animate_reconstruction(image)
show_all_images(image)


from PIL import Image
import sys

text = sys.argv[1]
im = Image.open(text)
width, height = im.size
pixels = im.load()

for i in range(int(im.size[0])) :
	for j in range(int(im.size[1])) :
		pixels[i,j] =  (tuple(int(t/2) for t in pixels[i,j]))

im.save('result.png')

from PIL import Image

im = Image.open('westbrook.jpg')
width, height = im.size
pixels = im.load()

for i in range(int(im.size[0])) :
	for j in range(int(im.size[1])) :
		pixels[i,j] =  (tuple(int(t/2) for t in pixels[i,j]))

im.save('result.jpg')
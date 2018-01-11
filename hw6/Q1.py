import numpy as np
from numpy import linalg as la
from skimage import io , transform
import time 
import sys

def read_data(dir_path):   
    faces = []
    for i in range(415) :
        im = io.imread(dir_path+'/'+str(i)+'.jpg')
        new_im = transform.resize(im,(300,300,3))
        faces.append(new_im)
    faces = np.array(faces)
    return faces

faces = read_data(sys.argv[1])
print ('faces : {}'.format(faces.shape))

###  mean face  ###
mu = np.mean(faces,axis = 0)
#mu1 = (mu*255).astype(np.uint8)
#io.imsave('Q1_res/mu.jpg',mu1)

###  caculate svd for faces-mean ###
start = time.time()
faces = faces.reshape(415,300*300*3)
flatten_mu = mu.reshape(300*300*3)
svd_face = (faces-flatten_mu).T
U,S,V = la.svd(svd_face,full_matrices=False)
print ('Caculate SVD in {}'.format(time.time() - start))

####  print rate of top 4 eigenvalue  ###
#print ('Top 4 :')
#for i in range(4):
#    print (S[i] / sum(S))
#print ('------')

###  save 10 eigen faces  ###
#for i in range(10):
#    M = U[:,i].reshape(300,300,3)
#    M = M-np.min(M)
#    M = M/np.max(M)
#    M = (M * 255).astype(np.uint8)
#    io.imsave('Q1_res/eigen_face'+str(i+1)+'.jpg',M)
    
###  reconstructe random 4 faces###
k = 200
for i in range(1):
    idx = np.random.randint(0,415)
    idx = int(sys.argv[2].split('.')[0])
    recon_face = np.zeros(300*300*3)
    for e in range(k) :
        recon_face += np.dot(faces[idx]-flatten_mu,U[:,e]) * U[:,e]
    recon_face = (recon_face+flatten_mu).reshape(300,300,3)  
    
    recon_face = recon_face-np.min(recon_face)
    recon_face = recon_face/np.max(recon_face)
    recon_face = (recon_face*255).astype(np.uint8)
    io.imsave('reconstruction.jpg',recon_face)


# In[ ]:





import scipy.io 
import matplotlib.pyplot as plt

matdata = scipy.io.loadmat("PATH")
gdt = matdata['gdt']
structure = matdata['gdt']['Structure']

#len(structure[0][0][0][0]) =200

matdata['gdr'][0][0]


gdr = matdata['gdr'][0]

RealizedSVP = 70.574725

TargetSVP = 70


for i in range(20):
    plt.imshow(structure[0][0][i*10,;,;])
    plt.show()
    if i>10:
        break




structure[0][0][0][0].shape = (200,0)

structure[0][0].shape = (200,200,200)




structure[0]
# array([array([[[0,0,0,...,0,0,0],
# [0,0,0,...,0,0,0],
# [0,0,0,...,0,0,0],
# ...,
# [1,1,1,...,2,0,0],
# [1,1,1,...,0,0,0],
# [1,1,1,...,0,0,0]],

# [[1,1,1,...,2,0,0],
#  [0,0,0,...,0,0,0],
# ...,
# [1,1,1,...,2,0,0],
# [1,1,1,...,0,0,0],
# [1,1,1,...,0,0,0]],

# ...


# [1,1,1,...,0,0,0]]],dtype=uint8)],dtype=object)



structure[0][0][i*10,;,;] 

figure 

200*200  particles



structure[0][0][0] 
array([[0,0,0,...,0,0,0],
[0,0,0,...,0,0,0],
[0,0,0,...,0,0,0],
...,
[1,1,1,...,2,0,0],
[1,1,1,...,0,0,0],
[1,1,1,...,0,0,0]], dtype=uint8)
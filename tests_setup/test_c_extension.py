import numpy as np
import cos_doubles

a = np.array([np.pi/3,np.pi/2,np.pi], dtype=np.double)
b = np.array([0,0,0], dtype=np.double)

cos_doubles.cos_db(a,b)
print b

c = cos_doubles.cos_db(a)
print c

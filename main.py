#1.1

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

X, Y = np.mgrid[-4:4:0.01, -4:4:0.01]
xs = tf.constant(X.astype(np.float32))
ys = tf.constant(Y.astype(np.float32))

tf.global_variables_initializer().run()

zs = tf.exp(-(xs**2 + ys**2)/2.0)
a = zs.eval()
plt.imshow(a)
plt.tight_layout()
plt.show()


#1.2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

X, Y = np.mgrid[-4:4:0.01, -4:4:0.01]
xs = tf.constant(X.astype(np.float32))
ys = tf.constant(Y.astype(np.float32))



tf.global_variables_initializer().run()

zs = tf.sin(X+Y)
a = zs.eval()
plt.imshow(a)
plt.tight_layout()
plt.show()

#1.3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

X, Y = np.mgrid[-4:4:0.01, -4:4:0.01]
xs = tf.constant(X.astype(np.float32))
ys = tf.constant(Y.astype(np.float32))



tf.global_variables_initializer().run()

zs = tf.sin(X+Y)
qs = tf.exp(-(xs**2 + ys**2)/2.0)
a = zs.eval()
b = qs.eval()
plt.imshow(a*b)
plt.tight_layout()
plt.show()

#2.1
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

Y, X = np.mgrid[-1.3:1.3:0.003, -2:1:0.003]
Z = X + 1j*Y
xs = tf.constant(Z.astype(np.complex64))
zs = tf.Variable(xs)
ns = tf.Variable(tf.zeros_like(xs, tf.float32))
tf.global_variables_initializer().run()

zs_ = zs*zs + xs
not_diverged = tf.abs(zs_) < 4
step = tf.group( zs.assign(zs_), ns.assign_add(tf.cast(not_diverged, tf.float32)) )

for i in range(200):
  step.run()

fig = plt.figure(figsize=(16,10))

def processFractal(a):
  a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
  img = np.concatenate([10+20*np.cos(a_cyclic),
  30+50*np.sin(a_cyclic),
  155-80*np.cos(a_cyclic)], 2)
  img[a==a.max()] = 0
  a = img
  a = np.uint8(np.clip(a, 0, 255))
  return a

plt.imshow(processFractal(ns.eval()))
plt.tight_layout(pad=0)
plt.show()

#2.2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

Y, X = np.mgrid[-1.3:0.3:0.002, -1:1:0.002]
Z = X + 1j*Y
xs = tf.constant(Z.astype(np.complex64))
zs = tf.Variable(xs)
ns = tf.Variable(tf.zeros_like(xs, tf.float32))
tf.global_variables_initializer().run()

zs_ = zs*zs + xs
not_diverged = tf.abs(zs_) < 4
step = tf.group( zs.assign(zs_), ns.assign_add(tf.cast(not_diverged, tf.float32)) )

for i in range(200):
  step.run()

fig = plt.figure(figsize=(16,10))

def processFractal(a):
  a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
  img = np.concatenate([10+20*np.cos(a_cyclic),
  30+50*np.sin(a_cyclic),
  155-80*np.cos(a_cyclic)], 2)
  img[a==a.max()] = 0
  a = img
  a = np.uint8(np.clip(a, 0, 255))
  return a

plt.imshow(processFractal(ns.eval()))
plt.tight_layout(pad=0)
plt.show()



#2.3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

Y, X = np.mgrid[-1.3:1.3:0.002, -2:3:0.002]
Z = X + 1j*Y
c = -0.8 + 1j*0.156
xs = tf.constant(Z.astype(np.complex64))
zs = tf.Variable(xs)
ns = tf.Variable(tf.zeros_like(xs, tf.float32))
tf.global_variables_initializer().run()

zs_ = zs*zs + c
not_diverged = tf.abs(zs_) < 4
step = tf.group( zs.assign(zs_), ns.assign_add(tf.cast(not_diverged, tf.float32)) )

for i in range(200):
  step.run()
  
 

fig = plt.figure(figsize=(16,10))

def processFractal(a):
  a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
  img = np.concatenate([10+20*np.cos(a_cyclic),
  30+50*np.sin(a_cyclic),
  155-80*np.cos(a_cyclic)], 2)
  img[a==a.max()] = 0
  a = img
  a = np.uint8(np.clip(a, 0, 255))
  return a

plt.imshow(processFractal(ns.eval()))
plt.tight_layout(pad=0)
plt.show()

sess.close()

#3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

Y, X = np.mgrid[-3:3:0.005, -3:3:0.005]
Z = X + 1j*Y
c = -0.5 + 1j*0.9
xs = tf.constant(Z.astype(np.complex64))
zs = tf.Variable(xs)
ns = tf.Variable(tf.zeros_like(xs, tf.float32))
tf.global_variables_initializer().run()


zs_ = tf.math.cos(zs)*tf.math.sin(zs) + xs
not_diverged = tf.abs(zs_) < 10
step = tf.group( zs.assign(zs_), ns.assign_add(tf.cast(not_diverged, tf.float32)) )

for i in range(200):
  step.run()
  
 

fig = plt.figure(figsize=(20,20))

def processFractal(a):
  a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
  img = np.concatenate([10+20*np.cos(a_cyclic),
  30+50*np.sin(a_cyclic),
  155-80*np.cos(a_cyclic)], 2)
  img[a==a.max()] = 0
  a = img
  a = np.uint8(np.clip(a, 0, 255))
  return a

plt.imshow(processFractal(ns.eval()))
plt.tight_layout(pad=0)
plt.show()

sess.close()




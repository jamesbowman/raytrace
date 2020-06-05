import Image
import numpy as np
import time
import numbers

class vec3():
    def __init__(self, x, y, z):
        (self.x, self.y, self.z) = (x, y, z)
    def __mul__(self, other):
        return vec3(self.x * other, self.y * other, self.z * other)
    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
    def __abs__(self):
        return self.dot(self)
    def norm(self):
        mag = np.sqrt(abs(self))
        return self * (1.0 / np.where(mag == 0, 1, mag))
    def components(self):
        return (self.x, self.y, self.z)
rgb = vec3

(w, h) = (512, 512)
x = np.tile(np.linspace(-1, 1, w), h)
y = np.repeat(np.linspace(-1, 1, h), w)

if 0:
    c = np.sqrt(.5)
    print 'c', c
    def norm(v):
        # v -= min(v)
        v /= max(v)
        return v
    gx = norm(np.exp(-(x*x) / (2 * c ** 2)))
    gy = norm(np.exp(-(y*y) / (2 * c ** 2)))
    g = gx * gy
else:
    g = ((x*x) + (y*y))
    if 1:
        g = g / 2
    else:
        g = np.sqrt(g) / 1.4
    g = 1 - g
print min(g), max(g)
print g[256]
c0 = rgb(1, 1, 0)
c1 = rgb(0, 0, 1)

color = (c0 * g) + c1 * (1 - g)

# color.y = np.where(g > 0.5, 0.0, 1.0)

rgb = [Image.fromarray((255 * c.reshape((h, w))).astype(np.uint8), "L") for c in color.components()]
Image.merge("RGB", rgb).save("fig.png")

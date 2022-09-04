from PIL import Image
from functools import reduce
import time

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
    def norm(self, np):
        mag = np.sqrt(abs(self))
        return self * (1.0 / np.where(mag == 0, 1, mag))
    def components(self):
        return (self.x, self.y, self.z)
rgb = vec3

(w, h) = (256, 256)         # Screen size
L = vec3(5, 5., -10)        # Point light position
E = vec3(0., 0.35, -1.)     # Eye position
FARAWAY = 1.0e39            # an implausibly huge distance

def raytrace(np, O, D, scene, bounce = 0):
    # O is the ray origin, D is the normalized ray direction
    # scene is a list of Sphere objects (see below)
    # bounce is the number of the bounce, starting at zero for camera rays

    distances = [s.intersect(np, O, D) for s in scene]
    nearest = reduce(np.minimum, distances)
    color = rgb(0, 0, 0)
    for (s, d) in zip(scene, distances):
        color += s.light(np, O, D, d, scene, bounce) * (nearest != FARAWAY) * (d == nearest)
    return color

class Sphere:
    def __init__(self, center, r, diffuse, mirror = 0.5):
        self.c = center
        self.r = r
        self.diffuse = diffuse
        self.mirror = mirror

    def intersect(self, np, O, D):
        b = 2 * D.dot(O - self.c)
        c = abs(self.c) + abs(O) - 2 * self.c.dot(O) - (self.r * self.r)
        disc = (b ** 2) - (4 * c)
        sq = np.sqrt(np.maximum(0, disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)

        pred = (disc > 0) & (h > 0)
        return np.where(pred, h, FARAWAY)

    def diffusecolor(self, M):
        return self.diffuse

    def light(self, np, O, D, d, scene, bounce):
        M = (O + D * d)                         # intersection point
        N = (M - self.c) * (1. / self.r)        # normal
        toL = (L - M).norm(np)                    # direction to light
        toO = (E - M).norm(np)                    # direction to ray origin
        nudged = M + N * .0001                  # M nudged to avoid itself

        # Shadow: find if the point is shadowed or not.
        light_distances = [s.intersect(np, nudged, toL) for s in scene]
        light_nearest = reduce(np.minimum, light_distances)
        seelight = light_distances[scene.index(self)] == light_nearest

        # Ambient
        color = rgb(0.05, 0.05, 0.05)

        # Lambert shading (diffuse)
        lv = np.maximum(N.dot(toL), 0)
        color += self.diffusecolor(M) * lv * seelight

        return color
        # Reflection
        if bounce < 2:
            rayD = (D - N * 2 * D.dot(N)).norm(np)
            color += raytrace(np, nudged, rayD, scene, bounce + 1) * self.mirror

        # Blinn-Phong shading (specular)
        phong = N.dot((toL + toO).norm(np))
        color += rgb(1, 1, 1) * np.power(np.clip(phong, 0, 1), 50) * seelight
        return color

class CheckeredSphere(Sphere):
    def diffusecolor(self, M):
        return self.diffuse
        checker = ((M.x * 2).astype(int) % 2) == ((M.z * 2).astype(int) % 2)
        return self.diffuse * checker

scene = [
    Sphere(vec3(.75, .1, 1.), .6, rgb(0, 0, 1)),
    Sphere(vec3(-.75, .1, 2.25), .6, rgb(.5, .223, .5)),
    Sphere(vec3(-2.75, .1, 3.5), .6, rgb(1., .572, .184)),
    CheckeredSphere(vec3(0,-99999.5, 0), 99999, rgb(.75, .75, .75), 0.25),
    ]
def evaluator(np, x, y):
    Q = vec3(x, y, 0)
    return raytrace(np, E, (Q - E).norm(np), scene)
import numpy as np

import numpy as mm

r = float(w) / h
# Screen coordinates: x0, y0, x1, y1.
S = (-1., 1. / r + .25, 1., -1. / r + .25)
x = mm.tile(np.linspace(S[0], S[2], w), h)
y = mm.repeat(np.linspace(S[1], S[3], h), w)

t0 = time.time()
color = evaluator(mm, x, y)
print(f"Took {(time.time() - t0):.3f} s")

frgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((h, w))).astype(np.uint8), "L") for c in color.components()]
Image.merge("RGB", frgb).save("rt5.png")

mem = set()

BIN=1500
UNA=10
LOG=32
MUX=32

class Mock:
    def __init__(self, n = 0):
        self.n = n
        mem.add(self)

    def __repr__(self):
        return f"<{self.n} operations>"

    def unop(self, other):
        return Mock(Mock.cost(self) + UNA)
    def binop(self, other, ocost = BIN):
        return Mock(Mock.cost(self) + Mock.cost(other) + ocost)

    def __add__(self, other):
        return self.binop(other)
    def __radd__(self, other):
        return self.binop(other)
    def __sub__(self, other):
        return self.binop(other)
    def __rsub__(self, other):
        return self.binop(other)
    def __mul__(self, other):
        return self.binop(other)
    def __rmul__(self, other):
        return self.binop(other)
    def __truediv__(self, other):
        return self.binop(other)
    def __pow__(self, other):
        return self.binop(other)
    def __rtruediv__(self, other):
        return self.binop(other)
    def __lt__(self, other):
        return self.binop(other)
    def __gt__(self, other):
        return self.binop(other)
    def __and__(self, other):
        return self.binop(other, LOG)
    def __neg__(self):
        return self.unop(self)

    @staticmethod
    def cost(c):
        if isinstance(c, Mock) and (c not in mem):
            return c.n
        else:
            return 0
    def costs(L):
        return sum([Mock.cost(c) for c in L])
        
    @staticmethod
    def sqrt(c):
        return Mock(Mock.cost(c) + 4*BIN)

    @staticmethod
    def where(a, b, c):
        return Mock(Mock.costs([a, b, c]) + MUX)

    @staticmethod
    def clip(a, b, c):
        return Mock(Mock.costs([a, b, c]) + 2*BIN+2*MUX)

    @staticmethod
    def maximum(a, b):
        return Mock.binop(a, b)

    @staticmethod
    def minimum(a, b):
        return Mock.binop(a, b)

    @staticmethod
    def power(a, b):
        return Mock.binop(a, b)

x = Mock()
y = Mock()
evaluator(Mock, x, y)
ops = sum([t.n for t in mem])
print(f"{ops=}")
sysfreq = 300
print(f"For {sysfreq} MHz device {ops/(sysfreq * 1e6):.3f} s")

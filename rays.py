from PIL import Image
import math
import numpy as np
import rt4
import time

rgb = rt4.rgb
vec3 = rt4.vec3

########################################################################

# Based on "JAVA REFERENCE IMPLEMENTATION OF IMPROVED NOISE - COPYRIGHT 2002 KEN PERLIN."
# http://mrl.nyu.edu/~perlin/noise/

p0 = [ 151,160,137,91,90,15,
   131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
   190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
   88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
   77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
   102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
   135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
   5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
   223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
   129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
   251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
   49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
   138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180] * 2

perm = np.array(p0).astype(np.uint8)
perm15 = perm & 15

def frac(x):
    return x - np.floor(x)

def lerp(t, a, b):
    return a + t * (b - a)

def mod256(x):
    q = x * (1.0 / 256.0)
    f = frac(q)
    return (f * 256).astype(int)

def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

#       u   v
# 0     x   y
# 4     x   z
# 8     y   z
# 12    y   x
# 14    y   x
def grad(prehash, x, y, z):
    h = np.take(perm15, prehash)
    u = np.where(h < 8, x, y)
    v = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, z))
    return np.where((h & 1) == 0, u, -u) + np.where((h & 2) == 0, v, -v)

def noise(p):
    X = np.floor(p.x)
    Y = np.floor(p.y)
    Z = np.floor(p.z)

    X = X.astype(np.uint8) & 0xff
    Y = Y.astype(np.uint8) & 0xff
    Z = Z.astype(np.uint8) & 0xff

    x = frac(p.x)
    y = frac(p.y)
    z = frac(p.z)

    u = fade(x)
    v = fade(y)
    w = fade(z)

    A  = np.take(perm, X) + Y
    AA = np.take(perm, A) + Z
    AB = np.take(perm, A + 1) + Z
    B  = np.take(perm, X+1) + Y
    BA = np.take(perm, B) + Z
    BB = np.take(perm, B + 1) + Z

    return lerp(w, lerp(v, lerp(u, grad(AA  , x  , y  , z   ),  # AND ADD
                                   grad(BA  , x-1, y  , z   )), # BLENDED
                           lerp(u, grad(AB  , x  , y-1, z   ),  # RESULTS
                                   grad(BB  , x-1, y-1, z   ))),# FROM  8
                   lerp(v, lerp(u, grad(AA+1, x  , y  , z-1 ),  # CORNERS
                                   grad(BA+1, x-1, y  , z-1 )), # OF CUBE
                           lerp(u, grad(AB+1, x  , y-1, z-1 ),
                                   grad(BB+1, x-1, y-1, z-1 ))));

def fBm(point, H, lacunarity = 2.0, octaves = 8):
    value = 0.0
    for i in range(octaves):
        exponent = math.pow(lacunarity, -i*H)
        value += noise(point) * exponent
        point *= lacunarity
    return value

def moonFunc(p):
    # range of H is 0.4 - 0.9
    #H = lerp(0.4, 0.9, 0.5 + 0.5 * noise(p + vec3(9,0,8)))
    if 1:
        t = fBm(p + vec3(1,3,2), 0.6, 2, 7)
        return 1.0 + t

def moonHeight(p):
    f = moonFunc(p)
    return np.minimum(lerp(f - 1, 0.4, 0.405), 0.4)

def computeNormal(p, n, o, r, func):
    u = n.cross(r).norm()
    v = u.cross(n)

    epsilon = 1.e-4
    u *= epsilon
    v *= epsilon

    a = func(p)
    au = func(p + u)
    av = func(p + v)

    du = (u + n * (au - a))
    dv = (v + n * (av - a))
    n = dv.cross(du).norm()
    return n

class CheckeredSphere(rt4.Sphere):
    diffuse = rgb(1,1,1)

    def normal(self, p):
        n = (p - self.c) * (1. / self.r)        # normal

        r = vec3(-1, 0, 0)
        u = n.cross(r).norm()
        v = u.cross(n)

        epsilon = 1.e-4
        u *= epsilon
        v *= epsilon

        func = moonHeight
        a = func(p)
        au = func(p + u)
        av = func(p + v)

        du = (u + n * (au - a))
        dv = (v + n * (av - a))
        n = dv.cross(du).norm()

        return n

    def diffusecolor(self, M):
        f = moonFunc(M)
        mare = (f < 1)
        l = np.where(mare, lerp(f, 0.5, 0.6), lerp(f - 1, 0.8, 1.0))
        return rgb(255 * l, 248 * l, 220 * l) * (1. / 255)

def saw(x, y, t):
    n = noise(vec3(x, y, frac(t)))
    return n * (1 - (2 * abs(0.5 - frac(t))))

def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)

if __name__ == '__main__':
    D = 637
    w = 4 * D
    h = 4 * D
    x = np.tile(np.linspace(0, 1, w), h)
    y = np.repeat(np.linspace(0, 1, h), w)

    vel = -0.9
    for f in range(1):
        # polar
        r = np.sqrt(x ** 2 + y ** 2)
        th = np.arctan2(y, x)
        th1 = np.fmod(3 * th / (0.5 * math.pi), 1)

        # a is alpha mask
        a = np.where((0.16 < th1) * (th1 < 0.66), 1.0, 0.0)

        # d is distance to edge
        d = np.minimum(abs(th1 - 0.16), abs(th1 - 0.66))

        # b is brightness
        b = np.clip((1 - 4 * d) ** 6, .5, 1)

        # falloff towards (0,0)
        falloff = smoothstep(np.clip(5 * (r - .2), 0, 1))

        l = falloff * (b * a)
        color = rgb(1, 1, 1) * l

        crgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((h, w))).astype(np.uint8), "L") for c in color.components()]
        Image.merge("RGB", crgb).resize((D, D), Image.BICUBIC).save("%04d.png" % (f + 1))

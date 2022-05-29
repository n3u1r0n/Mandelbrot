import numpy as np
import multiprocessing as mp
from PIL import Image
from time import time
import os
import warnings
warnings.filterwarnings("ignore")

def iterate(args):
  c, max_iterations = args
  z = c.copy()
  iterations = np.ndarray(c.shape, dtype = np.uint64)
  iterations.fill(0)
  smaller = np.where(z * z.conj() < 4, True, False)
  for iteration in range(max_iterations):
    z[smaller] *= z[smaller]
    z[smaller] += c[smaller]
    smaller = np.where(z * z.conj() < 4, True, False)
    iterations[smaller] += 1
  del c, z, smaller
  return iterations

def generate_c(re, im, ppu):
  re = np.linspace(*re, int((re[1] - re[0]) * ppu))
  im = np.linspace(*im, int((im[1] - im[0]) * ppu))
  c = np.ndarray((len(im), len(re)), dtype = np.complex256)
  for i in range(len(im)):
    c[i,:] = re + im[-1 - i] * 1j
  del re, im, ppu
  return c

def iterations_mandelbrot(re, im, ppu, max_iterations, threadcount):
  c = generate_c(re, im, ppu)
  pool = mp.Pool(threadcount)
  iterations = np.array(pool.map(iterate, [(c, max_iterations) for c in c]))
  pool.close()
  del pool, c, re, im, ppu, max_iterations, threadcount
  return iterations

def map_colours(args):
  iterations, colours, max_iterations = args
  black = int(colours[0][0:2], 16), int(colours[0][2:4], 16), int(colours[0][4:6], 16)
  colours = [(int(colour[0:2], 16), int(colour[2:4], 16), int(colour[4:6], 16)) for colour in colours[1:]]
  colourmap = []
  for iteration in iterations:
    if iteration == max_iterations:
      colourmap.append(black)
    else:
      colourmap.append(colours[abs(int(iteration % (2 * len(colours) - 3) - len(colours) + 1))])
  del iterations, colours, black, max_iterations, args
  return colourmap


def mandelbrot(image_name = False,
               show = True,
               re = (-2.25, 0.75),
               im = (-1.25, 1.25),
               ppu = 200,
               max_iterations = 200,
               threadcount = 8,
               colours = ["ffffff","ffc00d", "e8980c", "ff8300", "e85d0c", "ff430d"]):
  iterations = iterations_mandelbrot(re, im, ppu, max_iterations, threadcount)
  pool = mp.Pool(threadcount)
  image = np.array(pool.map(map_colours, [(iterations, colours, max_iterations) for iterations in iterations]), dtype = np.uint8)
  pool.close()
  del pool
  image = Image.fromarray(image)
  if show:
    image.show()
  if image_name:
    image.save(image_name + ".png", "png")
  return image

colour_palette = [["ffffff","ffc00d", "e8980c", "ff8300", "e85d0c", "ff430d"],
                  ["ffffff", "002400","273b09", "58641d", "7b904b", "98a050"]]

mandelbrot("test", ppu = 6000, max_iterations = 600, colours = colour_palette[1])
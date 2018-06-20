#! /usr/bin/env python

import numpy
from scipy.misc import imsave 



# Parameters...
dims = [768, 1024]



# First calculate the set of images required...
def step_sizes(size):
  """Returns the relevant step sizes for a given dimension size."""
  ret = [1]
  
  while (ret[-1] * 2) < size:
    ret.append(ret[-1] * 2)
    
  return ret



steps = [step_sizes(size) for size in dims]

print 'Horizontal steps =', steps[1]
print 'Vertical steps =', steps[0]
print '%i images to be generated.' % (2 * sum([len(s) for s in steps]))



# Now loop and generate each image in turn...
offsets = dict()
image_index = 0

for dim in [0,1]: # 0 for vertical, 1 for horizontal.
  for step in steps[dim]:
    offset = (0, step) if dim==0 else (step, 0)
    offsets[offset] = [None, None]

    image = numpy.zeros(dims, dtype=numpy.uint8)
    
    phase = numpy.r_[numpy.zeros(step, dtype=numpy.uint8), 255*numpy.ones(step, dtype=numpy.uint8)]
    count = (dims[dim] // phase.shape[0]) + 1
    extended = numpy.tile(phase, count)[:dims[dim]]
    
    if dim==0:
      image[:,:] = extended[:,numpy.newaxis]
    else:
      image[:,:] = extended[numpy.newaxis,:]
    
    fn = 'gray%04d.png' % image_index
    offsets[offset][1] = fn
    imsave(fn, image)
    image_index += 1
    
    fn = 'gray%04d.png' % image_index
    offsets[offset][0] = fn
    imsave(fn, 255-image)
    image_index += 1



# Write out an offset file, that records how to convert each image response into an image coordinate...
f = open('gray.json', 'w')
f.write('[\n')

for key in sorted(offsets.keys()):
  off, on = offsets[key]
  f.write(' {"offset.x" : %i, "offset.y" : %i, "off" : "%s", "on" : "%s"},\n' % (key[0], key[1], off, on))

f.write(']\n')
f.close()

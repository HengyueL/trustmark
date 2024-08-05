# Copyright 2023 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


from trustmark import TrustMark
from PIL import Image
from pathlib import Path
import math,random
import numpy as np


#EXAMPLE_FILE = 'ufo_240.jpg'     # JPEG example
EXAMPLE_FILE = 'bfly_rgba.png'   # Transparent PNG example

# Available modes: C=compact, Q=quality, B=base
MODE="C"
tm=TrustMark(verbose=True, model_type=MODE, use_ECC=False)

# encoding example
cover = Image.open(EXAMPLE_FILE)
rgb=cover.convert('RGB')
has_alpha=cover.mode== 'RGBA'
if (has_alpha):
  alpha=cover.split()[-1]

random.seed(1234)
capacity=tm.schemaCapacity()
bitstring=''.join([random.choice(['0', '1']) for _ in range(capacity)])

print("Check input bitsring: ", bitstring)
encoded=tm.encode(rgb, bitstring, MODE='binary')

if (has_alpha):
  encoded.putalpha(alpha)
outfile=Path(EXAMPLE_FILE).stem+'_'+MODE+'.png'
encoded.save(outfile, exif=cover.info.get('exif'), icc_profile=cover.info.get('icc_profile'), dpi=cover.info.get('dpi'))

# decoding example
stego = Image.open(outfile).convert('RGB')
wm_secret, wm_present, wm_schema = tm.decode(stego, MODE='binary')
if wm_present:
  print(f'Extracted secret: {wm_secret} (schema {wm_schema})')
else:
  print('No watermark detected')

# psnr (quality, higher is better)
mse = np.mean(np.square(np.subtract(np.asarray(stego).astype(np.int16), np.asarray(rgb).astype(np.int16))))
if mse > 0:
  PIXEL_MAX = 255.0
  psnr= 20 * math.log10(PIXEL_MAX) - 10 * math.log10(mse)
  print('PSNR = %f' % psnr)

# Compute bitwise acc.
bit_len = len(bitstring)
ba = []
for i in range(bit_len):
  e = bitstring[i]
  d = wm_secret[i]
  if e == d:
    ba.append(1)
  else:
    ba.append(0)
print("Bitwise acc. ", np.mean(ba))

# # removal
# stego = Image.open(outfile).convert('RGB')
# im_recover = tm.remove_watermark(stego)
# im_recover.save('recovered.png', exif=stego.info.get('exif'), icc_profile=stego.info.get('icc_profile'), dpi=stego.info.get('dpi'))
# wm_secret, wm_present, wm_schema = tm.decode(im_recover)
# if wm_present:
#   print(f'Extracted secret: {wm_secret} (schema {wm_schema})')
# else:
#    print('No secret after removal')

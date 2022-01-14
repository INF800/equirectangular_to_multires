from __future__ import print_function

import imp
import requests
import time

import asyncio
# import aiohttp

import concurrent.futures
import threading

"""
installation:
-------------
    1. sudo su
    2. apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-numpy python3-pip python3-pil hugin-tools
    # note: remove to fight off pysh error
    # !3. pip install pyshtools
    3. pip uninstall pyshtools
usage:
------
  python equirectangular_to_multires.py {input_images_dir} {output_images_dir}
example:
--------
  python equirectangular_to_multires.py /home/aiml/rakesh/multires/pano_images/1 /home/aiml/rakesh/multires/multires_images/1 
"""

import argparse
from PIL import Image
import os
import sys
import math
import ast
from distutils.spawn import find_executable
import subprocess
import base64
import io
import numpy as np


def img2shtHash(img, lmax=5):
    '''
    Create spherical harmonic transform (SHT) hash preview.
    '''
    def encodeFloat(f, maxVal):
        return np.maximum(0, np.minimum(2 * maxVal, np.round(np.sign(f) * np.sqrt(np.abs(f)) * maxVal + maxVal))).astype(int)

    def encodeCoeff(r, g, b, maxVal):
        quantR = encodeFloat(r / maxVal, 9)
        quantG = encodeFloat(g / maxVal, 9)
        quantB = encodeFloat(b / maxVal, 9)
        return quantR * 19 ** 2 + quantG * 19 + quantB

    b83chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#$%*+,-.:;=?@[]^_{|}~"

    def b83encode(vals, length):
        result = ""
        for val in vals:
            for i in range(1, length + 1):
                result += b83chars[int(val // (83 ** (length - i))) % 83]
        return result

    # Calculate SHT coefficients
    r = pysh.expand.SHExpandDH(img[..., 0], sampling=2, lmax_calc=lmax)
    g = pysh.expand.SHExpandDH(img[..., 1], sampling=2, lmax_calc=lmax)
    b = pysh.expand.SHExpandDH(img[..., 2], sampling=2, lmax_calc=lmax)

    # Remove values above diagonal for both sine and cosine components
    # Also remove first row and column for sine component
    # These values are always zero
    r = np.append(r[0][np.tril_indices(lmax + 1)], r[1, 1:, 1:][np.tril_indices(lmax)])
    g = np.append(g[0][np.tril_indices(lmax + 1)], g[1, 1:, 1:][np.tril_indices(lmax)])
    b = np.append(b[0][np.tril_indices(lmax + 1)], b[1, 1:, 1:][np.tril_indices(lmax)])

    # Encode as string
    maxVal = np.max([np.max(r), np.max(b), np.max(g)])
    vals = encodeCoeff(r, g, b, maxVal).flatten()
    asstr = b83encode(vals, 2)
    lmaxStr = b83encode([lmax], 1)
    maxValStr = b83encode(encodeFloat([2 * maxVal / 255 - 1], 41), 1)
    return lmaxStr + maxValStr + asstr


def convert(inputFile, outputFolder):

    # Allow large images (this could lead to a denial of service attack if you're
    # running this script on user-submitted images.)
    Image.MAX_IMAGE_PIXELS = None

    # Find external programs
    try:
        nona = find_executable('nona')
    except KeyError:
        # Handle case of PATH not being set
        nona = None


    genPreview = False
    try:
        import pyshtools as pysh
        genPreview = True
    except:
        sys.stderr.write("Unable to import pyshtools. Not generating SHT preview.\n")


    # Subclass parser to add explaination for semi-option nona flag
    class GenParser(argparse.ArgumentParser):
        def error(self, message):
            if '--nona' in message:
                sys.stderr.write('''IMPORTANT: The location of the nona utility (from Hugin) must be specified
            with -n, since it was not found on the PATH!\n\n''')
            super(GenParser, self).error(message)

    # Parse input
    parser = GenParser(description='Generate a Pannellum multires tile set from a full or partial equirectangular or cylindrical panorama.',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('inputFile', metavar='INPUT'
    #                     help='panorama to be processed')
    parser.add_argument('-C', '--cylindrical', action='store_true',
                        help='input projection is cylindrical (default is equirectangular)')
    parser.add_argument('-H', '--haov', dest='haov', default=-1, type=float,
                        help='horizontal angle of view (defaults to 360.0 for full panorama)')
    parser.add_argument('-F', '--hfov', dest='hfov', default=100.0, type=float,
                        help='starting horizontal field of view (defaults to 100.0)')
    parser.add_argument('-V', '--vaov', dest='vaov', default=-1, type=float,
                        help='vertical angle of view (defaults to 180.0 for full panorama)') 
    parser.add_argument('-O', '--voffset', dest='vOffset', default=0.0, type=float,
                        help='starting pitch position (defaults to 0.0)')
    parser.add_argument('-e', '--horizon', dest='horizon', default=0.0, type=int,
                        help='offset of the horizon in pixels (negative if above middle, defaults to 0)')
    # parser.add_argument('-o', '--output', dest='output', default='./output',
    #                     help='output directory, optionally to be used as basePath (defaults to "./output")')
    parser.add_argument('-s', '--tilesize', dest='tileSize', default=512, type=int,
                        help='tile size in pixels')
    parser.add_argument('-f', '--fallbacksize', dest='fallbackSize', default=1024, type=int,
                        help='fallback tile size in pixels (defaults to 1024)')
    parser.add_argument('-c', '--cubesize', dest='cubeSize', default=0, type=int,
                        help='cube size in pixels, or 0 to retain all details')
    parser.add_argument('-b', '--backgroundcolor', dest='backgroundColor', default="[0.0, 0.0, 0.0]", type=str,
                        help='RGB triple of values [0, 1] defining background color shown past the edges of a partial panorama (defaults to "[0.0, 0.0, 0.0]")')
    parser.add_argument('-B', '--avoidbackground', action='store_true',
                        help='viewer should limit view to avoid showing background, so using --backgroundcolor is not needed')
    parser.add_argument('-a', '--autoload', action='store_true',
                        help='automatically load panorama in viewer')
    parser.add_argument('-q', '--quality', dest='quality', default=75, type=int,
                        help='output JPEG quality 0-100')
    parser.add_argument('--png', action='store_true',
                        help='output PNG tiles instead of JPEG tiles')
    parser.add_argument('--thumbnailsize', dest='thumbnailSize', default=0, type=int,
                        help='width of equirectangular thumbnail preview (defaults to no thumbnail; must be power of two; >512 not recommended)')
    parser.add_argument('-n', '--nona', default=nona, required=nona is None,
                        metavar='EXECUTABLE',
                        help='location of the nona executable to use')
    parser.add_argument('-G', '--gpu', action='store_true',
                        help='perform image remapping by nona on the GPU')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='debug mode (print status info and keep intermediate files)')
    args = parser.parse_args()


    # Check argument
    if args.thumbnailSize > 0:
        if args.thumbnailSize & (args.thumbnailSize - 1) != 0:
            print('Thumbnail size, if specified, must be a power of two')
            sys.exit(1)

    # Create output directory
    if os.path.exists(outputFolder):
        print('Output directory "' + outputFolder + '" already exists')
        if not args.debug:
            sys.exit(1)
    else:
        os.makedirs(outputFolder)

    # Process input image information
    print('Processing input image information...')
    origWidth, origHeight = Image.open(inputFile).size
    haov = args.haov
    if haov == -1:
        if args.cylindrical or float(origWidth) / origHeight == 2:
            print('Assuming --haov 360.0')
            haov = 360.0
        else:
            print('Unless given the --haov option, equirectangular input image must be a full (not partial) panorama!')
            sys.exit(1)
    vaov = args.vaov
    if vaov == -1:
        if args.cylindrical or float(origWidth) / origHeight == 2:
            print('Assuming --vaov 180.0')
            vaov = 180.0
        else:
            print('Unless given the --vaov option, equirectangular input image must be a full (not partial) panorama!')
            sys.exit(1)
    if args.cubeSize != 0:
        cubeSize = args.cubeSize
    else:
        cubeSize = 8 * int((360 / haov) * origWidth / math.pi / 8)
    tileSize = min(args.tileSize, cubeSize)
    levels = int(math.ceil(math.log(float(cubeSize) / tileSize, 2))) + 1
    if round(cubeSize / 2**(levels - 2)) == tileSize:
        levels -= 1  # Handle edge case
    origHeight = str(origHeight)
    origWidth = str(origWidth)
    origFilename = os.path.join(os.getcwd(), inputFile)
    extension = '.jpg'
    if args.png:
        extension = '.png'
    partialPano = True if args.haov != -1 and args.vaov != -1 else False
    colorList = ast.literal_eval(args.backgroundColor)
    colorTuple = (int(colorList[0]*255), int(colorList[1]*255), int(colorList[2]*255))

    if args.debug:
        print('maxLevel: '+ str(levels))
        print('tileResolution: '+ str(tileSize))
        print('cubeResolution: '+ str(cubeSize))

    # Generate PTO file for nona to generate cube faces
    # Face order: front, back, up, down, left, right
    faceLetters = ['f', 'b', 'u', 'd', 'l', 'r']
    projection = "f1" if args.cylindrical else "f4"
    pitch = 0
    text = []
    facestr = 'i a0 b0 c0 d0 e'+ str(args.horizon) +' '+ projection + ' h' + origHeight +' w'+ origWidth +' n"'+ origFilename +'" r0 v' + str(haov)
    text.append('p E0 R0 f0 h' + str(cubeSize) + ' w' + str(cubeSize) + ' n"TIFF_m" u0 v90')
    text.append('m g1 i0 m2 p0.00784314')
    text.append(facestr +' p' + str(pitch+ 0) +' y0'  )
    text.append(facestr +' p' + str(pitch+ 0) +' y180')
    text.append(facestr +' p' + str(pitch-90) +' y0'  )
    text.append(facestr +' p' + str(pitch+90) +' y0'  )
    text.append(facestr +' p' + str(pitch+ 0) +' y90' )
    text.append(facestr +' p' + str(pitch+ 0) +' y-90')
    text.append('v')
    text.append('*')
    text = '\n'.join(text)
    with open(os.path.join(outputFolder, 'cubic.pto'), 'w') as f:
        f.write(text)

    # Create cube faces
    print('Generating cube faces...')
    # args.gpu = True # monkey patch :C
    print("GPU ENABLED??", args.gpu)
    subprocess.check_call([args.nona, ('-g' if args.gpu else '-d') , '-o', os.path.join(outputFolder, 'face'), os.path.join(outputFolder, 'cubic.pto')])
    faces = ['face0000.tif', 'face0001.tif', 'face0002.tif', 'face0003.tif', 'face0004.tif', 'face0005.tif']

    # Generate tiles
    print('Generating tiles...')
    for f in range(0, 6):
        size = cubeSize
        faceExists = os.path.exists(os.path.join(outputFolder, faces[f]))
        if faceExists:
            face = Image.open(os.path.join(outputFolder, faces[f]))
            for level in range(levels, 0, -1):
                if not os.path.exists(os.path.join(outputFolder, str(level))):
                    os.makedirs(os.path.join(outputFolder, str(level)))
                tiles = int(math.ceil(float(size) / tileSize))
                if (level < levels):
                    face = face.resize([size, size], Image.ANTIALIAS)
                for i in range(0, tiles):
                    for j in range(0, tiles):
                        left = j * tileSize
                        upper = i * tileSize
                        right = min(j * args.tileSize + args.tileSize, size) # min(...) not really needed
                        lower = min(i * args.tileSize + args.tileSize, size) # min(...) not really needed
                        tile = face.crop([left, upper, right, lower])
                        if args.debug:
                            print('level: '+ str(level) + ' tiles: '+ str(tiles) + ' tileSize: ' + str(tileSize) + ' size: '+ str(size))
                            print('left: '+ str(left) + ' upper: '+ str(upper) + ' right: '+ str(right) + ' lower: '+ str(lower))
                        colors = tile.getcolors(1)
                        if not partialPano or colors == None or colors[0][1] != colorTuple:
                            # More than just one color (the background), i.e., non-empty tile
                            if tile.mode in ('RGBA', 'LA'):
                                background = Image.new(tile.mode[:-1], tile.size, colorTuple)
                                background.paste(tile, tile.split()[-1])
                                tile = background
                            tile.save(os.path.join(outputFolder, str(level), faceLetters[f] + str(i) + '_' + str(j) + extension), quality=args.quality)
                size = int(size / 2)

    # Generate fallback tiles
    print('Generating fallback tiles...')
    for f in range(0, 6):
        if not os.path.exists(os.path.join(outputFolder, 'fallback')):
            os.makedirs(os.path.join(outputFolder, 'fallback'))
        if os.path.exists(os.path.join(outputFolder, faces[f])):
            face = Image.open(os.path.join(outputFolder, faces[f]))
            if face.mode in ('RGBA', 'LA'):
                background = Image.new(face.mode[:-1], face.size, colorTuple)
                background.paste(face, face.split()[-1])
                face = background
            face = face.resize([args.fallbackSize, args.fallbackSize], Image.ANTIALIAS)
            face.save(os.path.join(outputFolder, 'fallback', faceLetters[f] + extension), quality = args.quality)

    # Clean up temporary files
    if not args.debug:
        os.remove(os.path.join(outputFolder, 'cubic.pto'))
        for face in faces:
            if os.path.exists(os.path.join(outputFolder, face)):
                os.remove(os.path.join(outputFolder, face))

    # Generate preview (but not for partial panoramas)
    if haov < 360 or vaov < 180:
        genPreview = False
    if genPreview:
        # Generate SHT-hash preview
        shtHash = img2shtHash(np.array(Image.open(inputFile).resize((1024, 512))))
    if args.thumbnailSize > 0:
        # Create low-resolution base64-encoded equirectangular preview image
        img = Image.open(inputFile)
        img = img.resize((args.thumbnailSize, args.thumbnailSize // 2))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=75, optimize=True)
        equiPreview = bytes('data:image/jpeg;base64,', encoding='utf-8')
        equiPreview += base64.b64encode(buf.getvalue())
        equiPreview = equiPreview.decode()

    # Generate config file
    text = []
    text.append('{')
    text.append('    "hfov": ' + str(args.hfov)+ ',')
    if haov < 360:
        text.append('    "haov": ' + str(haov)+ ',')
        text.append('    "minYaw": ' + str(-haov/2+0)+ ',')
        text.append('       "yaw": ' + str(-haov/2+args.hfov/2)+ ',')
        text.append('    "maxYaw": ' + str(+haov/2+0)+ ',')
    if vaov < 180:
        text.append('    "vaov": '    + str(vaov)+ ',')
        text.append('    "vOffset": ' + str(args.vOffset)+ ',')
        text.append('    "minPitch": ' + str(-vaov/2+args.vOffset)+ ',')
        text.append('       "pitch": ' + str(        args.vOffset)+ ',')
        text.append('    "maxPitch": ' + str(+vaov/2+args.vOffset)+ ',')
    if colorTuple != (0, 0, 0):
        text.append('    "backgroundColor": "' + args.backgroundColor+ '",')
    if args.avoidbackground and (haov < 360 or vaov < 180):
        text.append('    "avoidShowingBackground": true,')
    if args.autoload:
        text.append('    "autoLoad": true,')
    text.append('    "type": "multires",')
    text.append('    "multiRes": {')
    if genPreview:
        text.append('        "shtHash": "' + shtHash + '",')
    if args.thumbnailSize > 0:
        text.append('        "equirectangularThumbnail": "' + equiPreview + '",')
    text.append('        "path": "/%l/%s%y_%x",')
    text.append('        "fallbackPath": "/fallback/%s",')
    text.append('        "extension": "' + extension[1:] + '",')
    text.append('        "tileResolution": ' + str(tileSize) + ',')
    text.append('        "maxLevel": ' + str(levels) + ',')
    text.append('        "cubeResolution": ' + str(cubeSize))
    text.append('    }')
    text.append('}')
    text = '\n'.join(text)
    with open(os.path.join(outputFolder, 'config.json'), 'w') as f:
        f.write(text)




def generate_file(args):
    image_path, output_folder_base = args
    im_name = image_path.stem
    output_folder = output_folder_base / im_name
    convert(str(image_path), str(output_folder))


def generate_all_files_serial(files):
    for args in files:
        generate_file(args)


def generate_all_files_multithread(files, max_workers):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(generate_file, files)


def generate_all_files_multiproc(files, max_workers):
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(generate_file, files)


async def generate_all_files_async(files):
    tasks = []
    for args in files:
        task = asyncio.ensure_future(generate_file(args))
        tasks.append(task)
    await asyncio.gather(*tasks, return_exceptions=True)


def generate_all_files_asyncio(files):
    asyncio.get_event_loop().run_until_complete(generate_all_files_async(files))


def stats(file_count, duration, max_workers, show=False):
    sec_per_file = duration/file_count
    file_per_sec = file_count/duration

    min_per_file = (duration/60)/file_count
    file_per_min = file_count/(duration/60)

    if show:    
        print(f"{str('max_workers').ljust(20, '.')}",       f" {max_workers:.6f}".rjust(15, '.'),    " cores", sep='')
        print(f"{str('file_count').ljust(20, '.')}",        f" {file_count:.6f}".rjust(15, '.'),     " files", sep='')
        print(f"{str('duration').ljust(20, '.')}",          f" {duration:.6f}".rjust(15, '.'),       " seconds", sep='')
        print(f"{str('time for file').ljust(20, '.')}",     f" {sec_per_file:.6f}".rjust(15, '.'),   " sec/file", sep='')
        print(f"{str('files in second').ljust(20, '.')}",   f" {file_per_sec:.6f}".rjust(15, '.'),   " file/sec", sep='')
        print(f"{str('time for file').ljust(20, '.')}",     f" {min_per_file:.6f}".rjust(15, '.'),   " min/file", sep='')
        print(f"{str('files in minute').ljust(20, '.')}",   f" {file_per_min:.6f}".rjust(15, '.'),   " file/min", sep='')

    return dict(sec_per_file=sec_per_file,
                file_per_sec=file_per_sec,
                min_per_file=f"{min_per_file:.6f}", 
                file_per_min=f"{file_per_min:.6f}",)


#  taskset -c 0-31 python3 generate.py and softcode
#  ---------------------------------------------------------------
#            type  max_workers  file_count  total_duration_sec  total_duration_min  sec_per_file  file_per_sec min_per_file file_per_min
# 0        SERIAL           32           2           37.537695            0.625628     18.768847      0.053280     0.312814     3.196787
# 1  MULTI-THREAD           32           2           20.803699            0.346728     10.401849      0.096137     0.173364     5.768205
# 2    MULTI-PROC           32           2           20.940601            0.349010     10.470300      0.095508     0.174505     5.730495
#  ---------------------------------------------------------------
#  taskset -c 0,1,2,3 python3 generate.py and hardcode maxworker
#  ---------------------------------------------------------------
#            type  max_workers  file_count  total_duration_sec  total_duration_min  sec_per_file  file_per_sec min_per_file file_per_min
# 0        SERIAL            4           2           51.160787            0.852680     25.580394      0.039092     0.426340     2.345546
# 1  MULTI-THREAD            4           2           34.300037            0.571667     17.150018      0.058309     0.285834     3.498539
# 2    MULTI-PROC            4           2           34.445703            0.574095     17.222851      0.058062     0.287048     3.483744
#
#          type    cpu_count image_count  total_duration_sec  total_duration_min   sec_per_image  image_per_sec min_per_image image_per_min
#        SERIAL           32           2           37.537695            0.625628       18.768847       0.053280      0.312814      3.196787
#  MULTI-THREAD           32           2           20.803699            0.346728       10.401849       0.096137      0.173364      5.768205
#    MULTI-PROC           32           2           20.940601            0.349010       10.470300       0.095508      0.174505      5.730495
#        SERIAL            4           2           51.160787            0.852680       25.580394       0.039092      0.426340      2.345546
#  MULTI-THREAD            4           2           34.300037            0.571667       17.150018       0.058309      0.285834      3.498539
#    MULTI-PROC            4           2           34.445703            0.574095       17.222851       0.058062      0.287048      3.483744


if __name__ == "__main__":
    import sys
    import os
    import shutil
    import subprocess
    import pandas as pd
    from pathlib import Path
    
    data = dict(type=[],
                max_workers=[],
                file_count=[],
                total_duration_sec=[],
                total_duration_min=[],
                sec_per_file=[],
                file_per_sec=[],
                min_per_file=[], 
                file_per_min=[],)

    # max_workers = int(sys.argv[1])
    max_workers = os.cpu_count()
    # max_workers = 4

    print("="*os.get_terminal_size().columns)
    print("CPU INFORAMTION")
    print("="*os.get_terminal_size().columns)
    print((subprocess.check_output("lscpu", shell=True).strip()).decode())

    print("="*os.get_terminal_size().columns)
    print("STATISTICS")
    print("="*os.get_terminal_size().columns)

    input_images_dir = '/workspaces/equirectangular_to_multires/EQREC'
    output_dir = './tmp'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    image_paths = sorted([*Path(input_images_dir).glob("*.jpg")])[:16]
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    files = zip(image_paths, [output_dir]*len(image_paths))
    files = list(files)

    print("TOTAL_FILES:", len(files))


    start_time = time.time()
    generate_all_files_serial(files)
    duration = time.time() - start_time
    dic = stats(file_count=len(files), duration=duration, max_workers=max_workers, show=False)

    data['type'].append('SERIAL')
    data['max_workers'].append(max_workers)
    data['file_count'].append(len(files))
    data['total_duration_sec'].append(duration)
    data['total_duration_min'].append(duration/60)
    data['sec_per_file'].append(dic['sec_per_file'])
    data['file_per_sec'].append(dic['file_per_sec'])
    data['min_per_file'].append(dic['min_per_file'])
    data['file_per_min'].append(dic['file_per_min'])
    pd.DataFrame(data).to_csv(f"{str(len(data['type'])).zfill(3)}.csv", index=False)


    output_dir = './tmp'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)


    start_time = time.time()
    generate_all_files_multithread(files, max_workers=max_workers) 
    duration = time.time() - start_time
    dic = stats(file_count=len(files), duration=duration, max_workers=max_workers, show=False)

    data['type'].append('MULTI-THREAD')
    data['max_workers'].append(max_workers)
    data['file_count'].append(len(files))
    data['total_duration_sec'].append(duration)
    data['total_duration_min'].append(duration/60)
    data['sec_per_file'].append(dic['sec_per_file'])
    data['file_per_sec'].append(dic['file_per_sec'])
    data['min_per_file'].append(dic['min_per_file'])
    data['file_per_min'].append(dic['file_per_min'])
    pd.DataFrame(data).to_csv(f"{str(len(data['type'])).zfill(3)}.csv", index=False)


    output_dir = './tmp'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)


    start_time = time.time()
    generate_all_files_multiproc(files, max_workers=max_workers)
    duration = time.time() - start_time
    dic = stats(file_count=len(files), duration=duration, max_workers=max_workers, show=False)

    data['type'].append('MULTI-PROC')
    data['max_workers'].append(max_workers)
    data['file_count'].append(len(files))
    data['total_duration_sec'].append(duration)
    data['total_duration_min'].append(duration/60)
    data['sec_per_file'].append(dic['sec_per_file'])
    data['file_per_sec'].append(dic['file_per_sec'])
    data['min_per_file'].append(dic['min_per_file'])
    data['file_per_min'].append(dic['file_per_min'])
    print(pd.DataFrame(data))
    pd.DataFrame(data).to_csv(f"{str(len(data['type'])).zfill(3)}.csv", index=False)

    # output_dir = './tmp'
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)


    # start_time = time.time()
    # generate_all_files_asyncio(files)
    # duration = time.time() - start_time
    # dic = stats(file_count=len(files), duration=duration, max_workers=max_workers, show=False)

    # data['type'].append('ASYNC')
    # data['max_workers'].append(max_workers)
    # data['file_count'].append(len(files))
    # data['total_duration_sec'].append(duration)
    # data['total_duration_min'].append(duration/60)
    # data['sec_per_file'].append(dic['sec_per_file'])
    # data['file_per_sec'].append(dic['file_per_sec'])
    # data['min_per_file'].append(dic['min_per_file'])
    # data['file_per_min'].append(dic['file_per_min'])


    # print(pd.DataFrame(data))


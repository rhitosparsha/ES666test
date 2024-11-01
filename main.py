import pdb
import src
import glob
import importlib
import os
import cv2



### Change path to images here
path = 'Images{}*'.format(os.sep)  # Use os.sep, Windows, linux have different path delimiters
###

all_submissions = glob.glob('./src/*')
os.makedirs('./results/', exist_ok=True)
for idx,algo in enumerate(all_submissions):
    print('****************\tRunning Awesome Stitcher developed by: {}  | {} of {}\t********************'.format(algo.split(os.sep)[-1],idx,len(all_submissions)))
    try:
        module_name = '{}_{}'.format(algo.split(os.sep)[-1],'stitcher')
        filepath = '{}{}stitcher.py'.format( algo,os.sep,'stitcher.py')
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        ###
        focal_lengths ={'I1': None, 'I2': 800, 'I3': 650, 'I4': None, 'I5': None, 'I6' : 700}
        Flags = {'I1': 0, 'I2': 1, 'I3': 1, 'I4': 0, 'I5': 0, 'I6' : 1}
        for impaths in glob.glob(path):
            PanaromaStitcher = getattr(module, 'PanaromaStitcher')
            inst = PanaromaStitcher(image_files=impaths, focal_length=focal_lengths[impaths[-2:]], Flag=Flags[impaths[-2:]])
            print('\t\t Processing... {}'.format(impaths))
            stitched_image, homography_matrix_list = inst.stitch_images()

            outfile =  './results/{}/{}.png'.format(impaths.split(os.sep)[-1],spec.name)
            os.makedirs(os.path.dirname(outfile),exist_ok=True)
            cv2.imwrite(outfile,stitched_image)
            print(homography_matrix_list)
            print('Panaroma saved ... @ ./results/{}.png'.format(spec.name))
            print('\n\n')

    except Exception as e:
        print('Oh No! My implementation encountered this issue\n\t{}'.format(e))
        print('\n\n')

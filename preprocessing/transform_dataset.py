import sys
import glob
import gflags
import cv2
from scipy.ndimage import rotate


FLAGS = gflags.FLAGS

# Input flags
gflags.DEFINE_string('img_path', "./dataset", 'Path where images to be resized.')

gflags.DEFINE_string('img_type', ".png", 'Image file extension with the dot.')

gflags.DEFINE_float('size_factor', 0.5, 'Uniform resize factor to apply to images.')

            
def main(argv):
    # Utility main to load flags
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
      sys.exit(1)
      
    # main loop
    for filename in glob.iglob(FLAGS.img_path + '/**/*' + FLAGS.img_type, recursive=True):
        # Read image
        im = cv2.imread(filename)
        height, width = im.shape[:2]
        # Convert into grayscale
        gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        # Resize
        target_size = (int(FLAGS.size_factor*width), int(FLAGS.size_factor*height))
        gray_im = cv2.resize(gray_im, target_size)
        # Rotate 90 degrees if necessary
        if height > width:
            gray_im = rotate(gray_im, 90)
            
        # Save image
        cv2.imwrite(filename, gray_im)
        

if __name__ == "__main__":
    main(sys.argv)
    

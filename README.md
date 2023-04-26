# Coloring_Photos
Using Python and open Cv to Color Black and white photos based of the model created by Rich Zhang

Image Colorization using OpenCV and Deep Learning

This is a Python script that uses OpenCV and a pre-trained deep neural network to colorize a grayscale image. The script is based on the "Colorful Image Colorization" technique developed by Zhang et al. (2016) and uses the Caffe deep learning framework.
Prerequisites

To run this script, you need to have the following installed:

    Python 3.x
    OpenCV 4.x
    NumPy
    Caffe (you can download the pre-trained model and associated files from the links provided in the script)

Usage

To colorize a grayscale image, simply run the script and pass the path to the input image as an argument:

css

python colorize.py -i input_image.jpg

The script will display the original grayscale image and the colorized image side-by-side.
Credits

The pre-trained model and associated files used in this script were developed by Zhang et al. (2016) and can be downloaded from the following sources:

    colorization_deploy_v2.prototxt: https://github.com/richzhang/colorization/blob/master/colorization_deploy_v2.prototxt
    pts_in_hull.npy: https://github.com/richzhang/colorization/blob/master/pts_in_hull.npy
    colorization_release_v2.caffemodel: https://www.dropbox.com/s/dx0qvhhp5hbztyz/colorization_release_v2.caffemodel?dl=1

References

Zhang, R., Isola, P., & Efros, A. A. (2016). Colorful image colorization. In European Conference on Computer Vision (pp. 649-666). Springer, Cham.

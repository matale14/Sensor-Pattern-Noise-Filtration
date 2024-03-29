import os, numpy, PIL
from PIL import Image

def average_images(path):
    # Access all JPG files in directory
    allfiles=os.listdir(path)
    imlist=[filename for filename in allfiles if  filename[-4:] in [".jpg",".JPG"]]

    # Assuming all images are the same size, get dimensions of first image
    w,h=Image.open(imlist[0]).size
    N=len(imlist)

    # Create a numpy array of floats to store the average (assume RGB images)
    arr=numpy.zeros((h,w,3),numpy.float)

    # Build up average pixel intensities, casting each image as an array of floats
    for im in imlist:
        imarr=numpy.array(Image.open(im),dtype=numpy.float)
        arr=arr+imarr/N

    # Round values in array and cast as 8-bit integer
    arr=numpy.array(numpy.round(arr),dtype=numpy.uint8)

    # Generate, save and preview final image
    out=Image.fromarray(arr,mode="RGB")
    out.save("Average.png")
    out.show()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("path", help="python average_images.py Folder_Path")

    args = parser.parse_args()
    average_images(args.path)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from lesson_functions import *
from scipy.ndimage.measurements import label

import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip


def export_frame(img):
    global saveImage
    global current_frame_index
    if (saveImage is True):
        current_frame_index += 1
        imgPath = "images_2/image_" + str(current_frame_index) + ".jpg"
        mpimg.imsave(imgPath, img)
    return img

def process_frame(img, returnHeatmap=False):
    #datasetname = "svcClassifier_small.p"
    datasetname = "svcClassifier_full.p"
    out_img, box_list = find_cars_subsamples(img, datasetname)
    draw_img, heatmap = process_image(out_img, box_list)

    global saveImage
    global current_frame_index;
    if (saveImage is True):
        current_frame_index += 1
        imgPath = "images_1/image_" + str(current_frame_index) + ".jpg"
        #imgPath = "images_project/image_" + str(current_frame_index) + ".jpg"
        mpimg.imsave(imgPath, draw_img)

    if (returnHeatmap is True):
        return draw_img, heatmap
    else:
        return draw_img

saveImage = False
current_frame_index = 0
displayImage = False
if __name__ == "__main__":

    if (displayImage is True):        
        index = 1
        for i in range(39):
            name = "images_2/image_" + str(index+i) + ".jpg"
            print(name)
            img = mpimg.imread(name)
            drawImg, heatmap = process_frame(img, True);
            
            fig = plt.figure()
            plt.subplot(121)
            plt.imshow(drawImg)
            plt.title('Car Positions')
            plt.subplot(122)
            plt.imshow(heatmap, cmap='hot')
            plt.title('Heat Map')
            fig.tight_layout()
            if (saveImage is True):                
                name = "images_1_with_heatmap/image_" + str(index+i) + ".jpg"
                fig.savefig(name)
            #plt.show()

    else:
        #output = 'test_result.mp4'
        #clip = VideoFileClip("test_video.mp4")
        
        output = 'project_result.mp4'
        clip = VideoFileClip("project_video.mp4")#.subclip(25.0, 27.0)
        video_clip = clip.fl_image(process_frame)
        #video_clip = clip.fl_image(export_frame)
        video_clip.write_videofile(output, audio=False)
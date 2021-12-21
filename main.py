import os
from skimage import io
from nets.ssfs import SSFS_Fuse
import yolo

def main(input_dir, output_dir, obj_detection_dir):
    """
    Image Fusion
    :param input_dir: str, input dir with all images stores in one folder
    :param output_dir: str, output dir with all fused images
    :return:
    """
    print("--------------------------------------")
    print("*************IMAGE FUSION*************")
    print("--------------------------------------")
    ssfs = SSFS_Fuse()
    images_name = sorted(list({item[:-6] for item in os.listdir(input_dir)}))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for image_name in images_name:
        print("Fusing {}".format(image_name))
        img1 = io.imread(os.path.join(input_dir, image_name + "_1.png"))
        img2 = io.imread(os.path.join(input_dir, image_name + "_2.png"))
        fused = ssfs.fuse(img1, img2)
        io.imsave(os.path.join(output_dir, image_name + ".png"), fused)

    print("----------------------------------------")
    print("*************OBJECT DETECTION***********")
    print("----------------------------------------")
    # Detecting objects before image fusion
    print("Detecting objects before image fusion")
    yolo.run_object_detection_for_images(input_dir, obj_detection_dir)

    # Detecting objects after image fusion
    print("Detecting objects after image fusion")
    yolo.run_object_detection_for_images(output_dir, obj_detection_dir)



if __name__ == "__main__":
    input_folder_name = input("Enter input folder name : ")
    output_folder_name = input("Enter image fusion destination folder name : ")
    obj_detection_folder_name = input("Enter object detection destination folder name : ")
    input_dir = os.path.join(os.getcwd(), "data", input_folder_name)
    output_dir = os.path.join(os.getcwd(), "data", output_folder_name)
    obj_detection_dir = os.path.join(os.getcwd(), "data", obj_detection_folder_name)
    main(input_dir, output_dir, obj_detection_dir)


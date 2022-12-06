import os
import cv2

def crop_one_grape(img, bboxes_scaled, output_path):#裁剪葡萄圖片
    n = 0
    for i in bboxes_scaled:
        if(i[0] < 0):
            i[0] = 0
        if(i[1] < 0):
            i[1] = 0
        if(i[2] < 0):
            i[2] = 0
        if(i[3] < 0):
            i[3] = 0
        im = img[int(i[1]):int(i[3]),int(i[0]):int(i[2])]
        if not os.path.exists(output_path + "single"):
            os.mkdir(output_path + "single")
        save_path_file = os.path.join(output_path + "single" + "/" +str(n)+".jpg")
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path_file, im)
        n += 1
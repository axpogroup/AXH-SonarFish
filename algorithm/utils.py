import cv2 as cv


def get_elapsed_ms(start_tick_count):
    return int((cv.getTickCount() - start_tick_count) / cv.getTickFrequency() * 1000)


def resize_img(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    return cv.resize(img, dim, interpolation=cv.INTER_AREA)

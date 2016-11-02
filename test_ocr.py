import cv2
import numpy as np

def make_init(filename = '/home/krozin/Pictures/ocr/meter1.png'):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2,2), np.uint8)
    x1, y1, x2, y2 = 127, 322, 467, 422
    crop_img = gray[y1:y2, x1:x2]
    crop_img2 = img[y1:y2, x1:x2]
    return img, gray, kernel, crop_img, crop_img2


def get_binarization(img, addaptive=False, inverting=True, show=False, threshold=127):
    if addaptive:
        final = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 3)
    else:
        res, final = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    if inverting:
        invert_img = 255-final
    if show:
        if inverting:
            cv2.imshow('image', invert_img)
        else:
            cv2.imshow('image', final)
    if inverting:
        return invert_img
    return final


def get_conturs(binary_img, color_img, show=False, xmin=27, xmax=38):
    digests = {}
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        for i,j in enumerate(contours):
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            if xmin < w <= xmax and xmin < h <= xmax and x >=0 and y >= 0:
                digests.update({x: color_img[y:y+h, x:x+w]})
                if show:
                    cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 0+i*10, 255-i*10), 2)
                    cv2.imshow('image', color_img)
        #if show:
            #cv2.drawContours(color_img, contours, -1, (0, 255, 0), 3)
            #cv2.imshow('window title', color_img)
    #print sorted(digests.keys())
    if len(digests) == 1:
        return contours, digests.values()
    else:
        # do_reordering
        final_list = []
        contours, digests.values()
        for i in sorted(digests.keys()):
            final_list.append(digests[i])
        return contours, final_list



def get_morfology(img, flag = 0b1111, show = False):
    if flag & 0b0001:
        final = cv2.dilate(img, kernel, iterations=1)
    elif flag & 0b0010:
        final = cv2.erode(img, kernel, iterations = 1)
    elif flag & 0b0100:
        final = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif flag & 0b1000:
        final = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    if show:
        cv2.imshow('image', final)


def get_canny(img, show = False):
    final = cv2.Canny(img, 3, 30, apertureSize = 3)
    if show:
        cv2.imshow('image', final)


def get_lines(binary_img, origin_img, show=False):
    lines = cv2.HoughLinesP(binary_img, 1, np.pi/180, 2, minLineLength=80, maxLineGap=10)
    if lines.all():
        for x1,y1,x2,y2 in lines[0]:
            cv2.line(origin_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if show:
            cv2.imshow('houghlines',origin_img)


def make_resize(input_img, xsize=30.0, ysize=30.0):
    wscale = xsize / input_img.shape[1]
    hscale = ysize / input_img.shape[0]
    dim = (int(wscale * input_img.shape[1]), int(hscale * input_img.shape[0]))
    resized = cv2.resize(input_img, dim, interpolation=cv2.INTER_AREA)
    return resized


def get_meter_digest(filename, show=False, xsize=30.0, ysize=30.0):
    digit_imgs = []
    img, gray, kernel, crop_img, crop_img2 = make_init(filename)
    #inv_img = get_binarization(crop_img)

    """ hack """
    th3 = cv2.adaptiveThreshold(crop_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5)
    ret2, otsu = cv2.threshold(th3, 130, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv_img = 255 - otsu

    conturimg, digests = get_conturs(inv_img, crop_img2, xmin=20, xmax=40)
    list_digest = digests
    for num, i in enumerate(list_digest):
        i_gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        i_res = make_resize(i_gray)
        i_inv = get_binarization(i_res)
        if num >= 4:
            invert_img = 255 - i_inv
            i_inv = invert_img
        if show:
            cv2.imshow('image', i_inv)
            cv2.waitKey(0)
        digit_imgs.append(i_inv)

    return digit_imgs

def get_template(filename, show=False, xsize=30.0, ysize=30.0):
    num, i_inv = None, None
    img, gray, kernel, crop_img, crop_img2 = make_init(filename)
    inv_img = get_binarization(gray, threshold=127)
    conturimg, list_digest = get_conturs(inv_img, img, xmin=60, xmax=73)
    if list_digest:
        i_gray = cv2.cvtColor(list_digest[0], cv2.COLOR_BGR2GRAY)
        i_res = make_resize(i_gray)
        i_inv = get_binarization(i_res)

        if show:
            cv2.imshow('image', i_inv)
            cv2.waitKey(0)
        num = filename[-5]
    return (num, i_inv)


def do_validation(template, input_img, xsize=30.0, ysize=30.0, show=False):
    """
    :param templates: nampy matrix of digit - 0..9
    :param input:   object which has been get from miter -
                    we are assume that it digit but we dont know what kind of
    :return: digit or None
    """
    count = xsize*ysize
    if template.shape != (xsize, ysize):
        return -1
    elif input_img.shape != (xsize, ysize):
        return -2
    for i, row in enumerate(template):
        for j, cow in enumerate(row):
            if input_img[i][j] != template[i][j]:
                count -= 1
    return count

if __name__ == "__main__":
    num, digit_img = get_template(filename='meter1_digit2.png')
    num2, digit_img2 = get_template(filename='meter1_digit7.png')
    num3, digit_img3 = get_template(filename='meter1_digit5.png')
    templates = ((num, digit_img), (num2, digit_img2), (num3, digit_img3))

    miter_digits = get_meter_digest(filename='meter1.png')

    for n, t in templates:
        for c, md in enumerate(miter_digits):
            positiv = do_validation(t, md)
            value = "GOOD!" if ((100 - 100*positiv/900) < 22.0) else "BAD!"
            print "do validation of number={}, \n" \
                  "get digit from image from left to right. position={}\n" \
                  "validation = {}\n" \
                  "count pozitiv pizels={}\n" \
                  "% error recognition is {}\n" \
                  "threshold is {}" \
                  "\n++++++++++++++++++++++++++++++++++++++++++++++++\n".format(n, c, value, positiv, 100-100*positiv/900, 22.0)
            cv2.imshow('test', md)
            cv2.waitKey(0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



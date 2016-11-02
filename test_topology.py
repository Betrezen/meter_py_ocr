import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt

# ---------------  GENERAL ACTIONS  --------------------------------------------
def make_init(filename, bbox):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2,2), np.uint8)
    x1, y1, x2, y2 = bbox
    crop_img = gray[y1:y2, x1:x2]
    crop_img2 = img[y1:y2, x1:x2]
    return img, gray, kernel, crop_img, crop_img2


def do_threashhold(filename, bbox):
    img, gray, kernel, crop_img, crop_img2 = make_init(filename, bbox)
    invert_img = 255 - crop_img

    # global thresholding
    ret1, th1 = cv2.threshold(invert_img, 135, 255, cv2.THRESH_BINARY)
    th1_inv = 255 - th1

    # addaptive
    # th3 = cv2.adaptiveThreshold(crop_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5)
    # ret2, otsu = cv2.threshold(th3, 140, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # th2 = 255 - otsu

    # Otsu's thresholding after Gaussian filtering
    #blur = cv2.GaussianBlur(invert_img, (5, 5), 0)
    #ret3, th3 = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu's thresholding
    #ret4, th4 = cv2.threshold(invert_img, 130, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    #th5_inv = cv2.adaptiveThreshold(invert_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5)
    #th6_inv = cv2.adaptiveThreshold(invert_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)

    return th1, crop_img, crop_img2


def do_conturs(copy_th2, crop_img2):
    contours, hierarchy = cv2.findContours(copy_th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(crop_img2, contours, -1, (0, 255, 0), 3)
    #cv2.imshow('window title', crop_img2)
    xmin = 30
    xmax = 38
    if contours:
        for i,j in enumerate(contours):
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            if xmin < w <= xmax and xmin < h <= xmax and x >= 0 and y >= 0:
                cv2.rectangle(crop_img2, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.imshow('image', crop_img2)
        cv2.waitKey(0)


def do_histogram(crop_img):
    # http://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html
    # find histograme
    invert_img = 255 - crop_img
    hist, bins = np.histogram(invert_img.ravel(), 256, [0, 256])
    plt.hist(invert_img.ravel(), 256, [0, 256]);    plt.show()


# ---------------  DASH MODEL --------------------------------------------
def do_linerezation(bin_img, min_signal_value=255, mad=2):
    """ we have matrix
        line = (x,y,weight) -> *--- -> x,y= * and weight= ---
        mad= minimal allow distance - or how many pixel started from x,y have signal > min_signal
    """
    object_list=[]
    for y, i in enumerate(bin_img):
        line = []
        weight = 0
        for x, j in enumerate(i):
            if j >= min_signal_value:
                # let's put x,y position and looking for next points where signal > 0
                if not line:
                    line.append(x)
                    line.append(y)
                    weight = 1
                else:
                    weight += 1
            else:
                if line:
                    if weight >= mad: # good. we found the end of line. let's put distance
                        line.append(weight)
                        object_list.append(line)
                        line=[]; weight = 0
                    else: # let's drop this line because it is short and less then line_min_allow_distance
                        line = []; weight = 0
                else:
                    continue
    return object_list


def get_objects(img, list_lines=None, mad=2, macd=0):
    """ objects = [ [object1] [object2] ... [object_m] ]
        object1 = [line1, line2, ... line_n]
        objectm = [line_l, line_x, ... line_p]
        each object dash's (line's) model
        why some lines below to one object? because line_y and line_y+1 across by x ordinat.
           so, check_cross(line_y, line_y+1) shall return true!

        example: 7 1 - image
            -------    --
               ---    ---
              ---      --
             ---       --

        >>> obj=[]
        >>> l1=[1,2,3]
        >>> l2=[21,22,23]
        >>> ll=[l1,l2]
        >>> obj.append(ll)
        >>> obj
        [[[1, 2, 3], [21, 22, 23]]]
    """

    objects = []
    lines = do_linerezation(img, mad=mad) if not list_lines else list_lines
    levels = list(set([j[1] for j in lines]))
    level0_lines = [j for j in lines if j[1] == 0]

    # we have nothing. let's fill it by lines with y=0
    if not objects:
        for j in level0_lines:
            objects.append([j])

    for i in levels[1:]:
        # get all lines with y=i
        check_lines = [j for j in lines if j[1] == i]
        # let's try to figure out if any check_line belong to any object
        for line in check_lines:
            cross_flag = False
            for num_obj, obj in enumerate(objects):
                cross_obj_lines = [j for j in obj if j[1] == (i-1) and check_cross(line, j, macd=macd)]
                if cross_obj_lines:
                    #print "line={}, obj_lines={}".format(line, cross_obj_lines)
                    obj.append(line)
                    cross_flag = True
            if not cross_flag:
                # create new object because we went through all objects and didn't find crossing
                objects.append([line])
    return objects


def check_cross(line1, line2, mad=2, macd=1):
    """ we should find several cases if line1 and line2 have cross by X ordinat
        line = (x,y,weight) -> *--- -> x,y= * and weight= ---
        x1 ---------- x11 -> min distance x2-x1 = weight; weight >= min allow distance. MAD
        x1----------x11        we have cross here by X ordinat.
             x2----------x22   min allow cross distance = (x22 - x11).                  MACD

        case a                 if x1<=x2 and x11<=x22 and x11>x2 and (x11-x2)>=MACD
          x1----------x11
               x2-----------x22

        case b                 if x1>=x2 and x11>=x22 and x22>x1 and (x22-x1)>=MACD
               x1----------x11
          x2-----------x22

        case c                 if x1>=x2 and x11<=x22 and (x11-x1)>=MACD
             x1---x11
          x2-----------x22

        case d                 if x1<=x2 and x22<=x11 and (x22-x2)>=MACD
          x1----------x11
             x2---x22

        case e                 if x1<=x2 and x11==x2. That shall be covered by a/b if macd=0
          x1----------x11
                      x2-----------x22
    """
    x1, x11 = line1[0], line1[0]+line1[2]
    x2, x22 = line2[0], line2[0]+line2[2]
    #print x1,x11, x2, x22, mad, macd

    if x11-x1 < mad or x22-x2 < mad:
        #print "case_00"
        return None
    if x1 <= x2 and x11 <= x22 and x11 >= x2 and (x11-x2) >= macd:
        #print "case_a"
        return True
    elif x1 >= x2 and x11 >= x22 and x22 >= x1 and (x22-x1) >= macd:
        #print "case_b"
        return True
    elif x1 >= x2 and x11 <= x22 and (x11-x1) >= macd:
        #print "case_c"
        return True
    elif x1 <= x2 and x11 >= x22 and (x22-x2) >= macd:
        #print "case_d"
        return True
    #print "case_0"
    return False


def check_cross_list(l1, l2, mad=2, macd=0):
    for i in l1:
        for j in l2:
            if check_cross(i, j, mad, macd):
                #print "CROSSING!!! l1={}, l2={}".format(l1, l2)
                return True, l1, l2
    return None


def check_cross_objs(obj1, obj2):
    # if we dont have lines on the same y for both objects
    # or we dont have lines on y-1 and y for both objects
    # then dont need to do analysis because nothing can be compare
    # if we don't have the same lines in both objects then need find cross
    levels1 = sorted(list(set([j[1] for j in obj1])))
    levels2 = sorted(list(set([j[1] for j in obj2])))
    levels = list(set(levels1 + levels2))
    union_levels = set(levels1) & set(levels2)

    lines1 = set(['{}_{}_{}'.format(i[0], i[1], i[2]) for i in obj1])
    lines2 = set(['{}_{}_{}'.format(i[0], i[1], i[2]) for i in obj2])
    joint_lines = lines1 | lines2
    union_lines = lines1 & lines2

    if not union_levels:
        if (levels2[0] > levels1[-1] and not (levels2[0] - levels1[-1]) == 1) or \
           (levels1[0] > levels2[-1] and not (levels1[0] - levels2[-1]) == 1):
            #print "Levels diff more > 1\n" \
            #      "level1={}\nlevel2={}\n" \
            #      "==================================".format(levels1, levels2)
            return 0, 0
    elif union_lines:
        #print "The same line(s)\n" \
        #      "lines1 & lines2 ={}\n" \
        #      "lines1 | lines2={}\n" \
        #      "==================================".format(union_lines, joint_lines)
        return 1, joint_lines
    else: # let's compare
        print("Levels are OK. check: any line obj1 cross any line of obj2.\n" \
              "level1={}\nlevel2={}\n" \
              "lines1={}\nlines2={}\n" \
              "==================================".format(levels1, levels2, lines1, lines2))
        cross = -1
        for i in levels:
            lobj1 = [j for j in obj1 if j[1] == i]
            lobj2 = [j for j in obj2 if j[1] == i]
            lobj1_up = [j for j in obj1 if j[1] == (i + 1)]
            lobj2_up = [j for j in obj2 if j[1] == (i + 1)]
            lobj1_down = [j for j in obj1 if j[1] == (i - 1)]
            lobj2_down = [j for j in obj2 if j[1] == (i - 1)]
            if check_cross_list(lobj1, lobj2) or check_cross_list(lobj2, lobj1):
                cross = 10
            elif check_cross_list(lobj1, lobj2_up):
                cross = 11
            elif check_cross_list(lobj1, lobj2_down):
                cross = 12
            elif check_cross_list(lobj2, lobj1_up):
                cross = 13
            elif check_cross_list(lobj2, lobj1_down):
                cross = 14
            if cross > 0:
                return cross, joint_lines
    return -100, 0


def do_combination(in_objects):
    # let's combine objects if line belong to both
    # and let's remove duplicates for objects who includes another obj

    objects = copy.copy(in_objects)
    obj_count = len(objects)
    print ("objects_count={}".format(obj_count))
    for num, obj in enumerate(objects[:-1]):
        snum = num
        robjs = objects[(num+1):]
        for robj in robjs:
            snum += 1
            res = check_cross_objs(obj, robj)
            if res[0] > 0:
                obj += robj
                #joint_lines = set(['{}_{}_{}'.format(i[0], i[1], i[2]) for i in obj])
                #tmp = [map(int, i.split('_')) for i in joint_lines]
                #obj = tmp
                objects.remove(robj)
                prt = "->SAME LINE(S) FOUND!!!\n" if res[0] == 1 else "->WE FOUND CROSS {}!!!\n".format(res[0])
                print (prt)
            else:
                #print "->WE DIDN'T FIND CROSS!!!\n"
                pass
    return objects


def get_bbox(obj):
    """ we have dash model and we need return back bbox around of this model """
    x_values = [i[0] for i in obj]
    y_values = [i[1] for i in obj]
    x_max_values = [i[0] + i[2] for i in obj]
    x_min = min(x_values)
    y_min = min(y_values)
    x_max = max(x_max_values)
    y_max = max(y_values)
    return (x_min, y_min, x_max, y_max)


def do_filtering(objects, threshold_min=2, threshold_max=10000, bbox=None):
    """ we should exclude objects where count of lines less threshold """

    new_objects = []
    for obj in objects:
        box_coord = get_bbox(obj)
        if threshold_min < len(obj) < threshold_max:
            if bbox:
                x_len = box_coord[2]-box_coord[0]
                y_len = box_coord[3]-box_coord[1]
                bbox_min = bbox[0]
                bbox_max = bbox[1]
                if bbox_min < x_len < (bbox_max+1) and \
                   bbox_min < y_len < (bbox_max+1):
                    new_objects.append(obj)
            else:
                new_objects.append(obj)
    return new_objects


# ---------------------  DRAW something ----------------------------
def draw_lines(img, lines):
    for line in lines:
        x,y, x2, y2 = line[0], line[1], line[0]+line[2], line[1]
        # Draw a dash's red line with thickness of 1 px
        cv2.line(img, (x, y), (x2, y2), (0, 255, 0), 1)
    cv2.imshow('dash_model', img)
    cv2.waitKey(0)

def draw_dash_object(img, obj, color=(0, 255, 0)):
    for line in obj:
        x,y, x2, y2 = line[0], line[1], line[0]+line[2], line[1]
        # Draw a dash's green line with thickness of 1 px
        cv2.line(img, (x, y), (x2, y2), color, 1)
    cv2.imshow('dash_model', img)
    cv2.waitKey(0)

def draw_dash_objects(img, objects):
    i = 0
    for num, obj in enumerate(objects):
        #print "object: len={} index={}\n" \
        #      "dash_model={}\n" \
        #      "===================================\n".format(len(obj), num, obj)
        for line in obj:
            x,y, x2, y2 = line[0], line[1], line[0]+line[2], line[1]
            # Draw a dash's green line with thickness of 1 px
            cv2.line(img, (x, y), (x2, y2), (0, 255-i*20, i*20), 1)
        i += 1
        cv2.imshow('dash_model', img)
        cv2.waitKey(0)

def draw_dash_bbox_objects(img, objects):
    for obj in objects:
        x,y, x2,y2 = get_bbox(obj)
        cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 1)
        cv2.imshow('dash_bbox_model', img)
        cv2.waitKey(0)


def run():
    img, gray, kernel, crop_img_, crop_img2_ = make_init('/home/krozin/Pictures/ocr/meter1.png',
                                                         bbox=(160, 347, 345, 395))
    # cv2.imshow('test', crop_img2_)
    # cv2.waitKey(0)

    th2, crop_img, crop_img2 = do_threashhold(
        '/home/krozin/Pictures/ocr/meter1.png', bbox=(160, 347, 345, 395))
    # cv2.imshow('test', th2)
    # cv2.waitKey(0)

    lines = do_linerezation(th2)
    # draw_lines(crop_img2, lines)

    objs = get_objects(th2, lines)
    #draw_dash_objects(crop_img2, objs)

    o_objs = do_combination(objs)
    n_objs = do_filtering(o_objs, bbox=[20, 38])
    draw_dash_objects(crop_img2, n_objs)
    draw_dash_bbox_objects(crop_img2, n_objs)

    # do_conturs(copy_th2, crop_img2)
    # do_conturs(copy_th2_inv, crop_img2)
    # do_histogram(crop_img)

    cv2.destroyAllWindows()


def _debug():

    def lines2obj(joint_lines):
        return [map(int, i.split('_')) for i in joint_lines]

    img, gray, kernel, crop_img, crop_img2 = make_init('/home/krozin/Pictures/ocr/meter1.png',
                                                         bbox=(160, 347, 345, 395))

    obj_2_first = [[14, 4, 3], [12, 5, 11], [10, 6, 16], [9, 7, 20], [7, 8, 23], [5, 9, 26], [4, 10, 28], [4, 11, 8], [24, 11, 8], [3, 12, 5], [26, 12, 7], [5, 13, 2], [28, 13, 5], [28, 14, 5], [29, 15, 4], [29, 16, 4], [29, 17, 4], [29, 18, 4], [29, 19, 4], [28, 20, 4], [28, 21, 4], [25, 22, 7], [22, 23, 8], [17, 24, 13], [14, 25, 14], [12, 26, 14], [10, 27, 11], [9, 28, 7], [9, 29, 4], [7, 30, 4], [7, 31, 3], [6, 32, 3], [5, 33, 3], [5, 34, 4], [4, 35, 5], [4, 36, 29], [3, 37, 31], [3, 38, 31], [3, 39, 30]]
    obj_2_second = [[109, 7, 12], [107, 8, 16], [104, 9, 22], [103, 10, 24], [102, 11, 7], [121, 11, 7], [101, 12, 5], [123, 12, 6], [100, 13, 5], [125, 13, 4], [101, 14, 2], [126, 14, 3], [126, 15, 4], [126, 16, 4], [127, 17, 3], [126, 18, 4], [126, 19, 4], [126, 20, 3], [126, 21, 3], [125, 22, 3], [124, 23, 4], [120, 24, 7], [117, 25, 9], [114, 26, 10], [111, 27, 10], [108, 28, 10], [106, 29, 7], [105, 30, 5], [104, 31, 4], [103, 32, 3], [102, 33, 3], [101, 34, 4], [101, 35, 3], [101, 36, 3], [100, 37, 4], [100, 38, 29], [100, 39, 30], [100, 40, 30], [111, 41, 2]]
    obj_7 = [[148, 7, 30], [147, 8, 31], [147, 9, 31], [147, 10, 5], [170, 10, 7], [147, 11, 4], [172, 11, 5], [147, 12, 4], [172, 12, 4], [148, 13, 2], [172, 13, 4], [172, 14, 3], [171, 15, 4], [171, 16, 3], [170, 17, 3], [170, 18, 3], [169, 19, 3], [169, 20, 3], [168, 21, 3], [168, 22, 3], [167, 23, 3], [167, 24, 3], [167, 25, 3], [166, 26, 3], [166, 27, 3], [165, 28, 3], [165, 29, 2], [164, 30, 3], [163, 31, 4], [163, 32, 3], [162, 33, 4], [162, 34, 3], [161, 35, 4], [161, 36, 3], [160, 37, 4], [160, 38, 3], [159, 39, 4], [159, 40, 3]]
    obj_5_1 = [[51, 5, 29], [51, 6, 30], [51, 7, 30], [51, 8, 30], [51, 9, 28], [51, 10, 5], [51, 11, 5], [51, 12, 5], [51, 13, 5], [51, 14, 4], [52, 15, 3], [52, 16, 3], [52, 17, 3], [52, 18, 3], [52, 19, 3], [52, 20, 3], [52, 21, 24], [52, 22, 26], [52, 23, 9], [72, 23, 7], [52, 24, 6], [74, 24, 5], [54, 25, 2], [76, 25, 4], [77, 26, 3], [78, 27, 3], [78, 28, 3], [78, 29, 3], [78, 30, 3], [78, 31, 3], [77, 32, 3], [76, 33, 4], [74, 34, 6], [72, 35, 7], [54, 36, 24], [56, 37, 20], [58, 38, 16], [61, 39, 10]]
    obj_5_2 = [[62, 19, 8], [58, 20, 16], [52, 21, 24], [52, 22, 26], [52, 23, 9], [72, 23, 7], [52, 24, 6], [74, 24, 5], [54, 25, 2], [76, 25, 4], [77, 26, 3], [78, 27, 3], [78, 28, 3], [78, 29, 3], [78, 30, 3], [78, 31, 3], [77, 32, 3], [76, 33, 4], [74, 34, 6], [72, 35, 7], [54, 36, 24], [56, 37, 20], [58, 38, 16], [61, 39, 10]]
    obj_5_3 = [[52, 33, 4], [51, 34, 6], [52, 35, 7], [54, 36, 24], [56, 37, 20], [58, 38, 16], [61, 39, 10]]

    levels1 = sorted(list(set([j[1] for j in obj_5_1])))
    levels2 = sorted(list(set([j[1] for j in obj_5_2])))
    levels3 = sorted(list(set([j[1] for j in obj_5_3])))
    levels = list(set(levels1 + levels2 + levels3))

    #print "level={}".format(levels1)
    #print "level={}".format(levels2)
    #print "level={}".format(levels3)

    #print "l1&l2={}; \nl1&l3={}; \nl2&l3={}".format(
    #    sorted(set(levels1) & set(levels2)),
    #    sorted(set(levels1) & set(levels3)),
    #    sorted(set(levels2) & set(levels3)))

    combine = False
    obj = obj_5_1
    robj = obj_5_2
    l1=l2=None

    res = check_cross_objs(obj, robj)
    if res[0] > 0:
        tmp_obj = lines2obj(res[1])
        res2 = check_cross_objs(tmp_obj, obj_5_3)
        draw_dash_object(crop_img2, tmp_obj)

        if res2[0] > 0:
            tmp_obj2 = lines2obj(res2[1])
            draw_dash_object(crop_img2, tmp_obj2, color=(255,0,0))


    '''
    for i in levels:
        lobj1 = [j for j in obj if j[1] == i]
        lobj2 = [j for j in robj if j[1] == i]
        lobj1_up = [j for j in obj if j[1] == (i + 1)]
        lobj2_up = [j for j in robj if j[1] == (i + 1)]
        lobj1_down = [j for j in obj if j[1] == (i - 1)]
        lobj2_down = [j for j in robj if j[1] == (i - 1)]

        res1 = check_cross_list(lobj1, lobj2)
        res2 = check_cross_list(lobj1, lobj2_up)
        res3 = check_cross_list(lobj1, lobj2_down)
        res4 = check_cross_list(lobj2, lobj1_up)
        res5 = check_cross_list(lobj2, lobj1_down)

        if res1:
            l1, l2 = res1[1], res1[2]
        elif res2:
            l1, l2 = res2[1], res2[2]
        elif res3:
            l1, l2 = res3[1], res3[2]
        elif res4:
            l1, l2 = res4[1], res4[2]
        elif res5:
            l1, l2 = res5[1], res5[2]

        if res1 or res2 or res3 or res4 or res5:
            combine = True
        if combine:
            print "\nWE FOUND CROSS!!!\n"
            break
    else:
        print "\nWE DIDN'T FIND CROSS!!!\n"
    print l1
    print l2

    if l1 and l2:
        l1 = l1[0]
        l2 = l2[0]
        x, y, x2, y2 = l1[0], l1[1], l1[0] + l1[2], l1[1]
        x3, y3, x4, y4 = l2[0], l2[1], l2[0] + l2[2], l2[1]
        cv2.line(crop_img2, (x, y), (x2, y2), (255, 255, 255), 1)
        cv2.imshow('cross', crop_img2)
        cv2.waitKey(0)
        cv2.line(crop_img2, (x3, y3), (x4, y4), (255, 255, 255), 1)
        cv2.imshow('cross', crop_img2)
        cv2.waitKey(0)
    '''

    # draw_dash_object(crop_img2, obj_2_first)
    # draw_dash_object(crop_img2, obj_2_second)
    # draw_dash_object(crop_img2, obj_7)
    #draw_dash_object(crop_img2, obj_5_1, color=(0, 255, 0))
    #draw_dash_object(crop_img2, obj_5_2, color=(255, 0, 0))
    #draw_dash_object(crop_img2, obj_5_3, color=(0, 0, 255))

if __name__ == "__main__":
    #_debug()
    run()


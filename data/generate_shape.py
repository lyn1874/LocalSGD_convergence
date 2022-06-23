"""
Created on 16:13 at 09/12/2021
@author: bo
Create the toy shape dataset
"""
import numpy as np
import os
import math
import shutil
from PIL import Image, ImageDraw


def draw_rectangle(image_shape, b_c, f_c, loc, previous_im=[]):
    """Args:
    image_shape: tuple (h, w)
    b_c: tuple, (r, g, b) the background color
    f_c: tuple, (r, g, b) the foreground color
    loc: [x0, y0, x1, y1], x0 is along the column and y0 is along the row, x0<x1, y0<y1
    -> right and down direction are for x1 and y1
    previous_im: another image
    """
    if not previous_im:
        previous_im = Image.new("RGB", image_shape, b_c)
    draw = ImageDraw.Draw(previous_im)
    draw.rectangle(loc, fill=f_c, outline=f_c)
    return previous_im


def draw_triangle(image_shape, b_c, f_c, loc, previous_im=[]):
    if not previous_im:
        previous_im = Image.new("RGB", image_shape, b_c)
    draw = ImageDraw.Draw(previous_im)
    draw.polygon(loc, fill=f_c, outline=f_c)
    return previous_im


def draw_circle(image_shape, b_c, f_c, loc, previous_im=[]):
    if not previous_im:
        previous_im = Image.new("RGB", image_shape, b_c)
    draw = ImageDraw.Draw(previous_im)
    draw.ellipse(loc, fill=f_c, outline=f_c)
    return previous_im


def draw_polygon(image_shape, b_c, f_c, side, x_scale, y_scale, previous_im=[]):
    if not previous_im:
        previous_im = Image.new("RGB", image_shape, b_c)
    draw = ImageDraw.Draw(previous_im)
    x_enlarge = np.ones([side, 2]) * 15
    loc = [((math.cos(th) + 1) * x_enlarge[j_iter, 0] + x_scale - 10,
            (math.sin(th) + 1) * x_enlarge[j_iter, 1] + y_scale - 10) for j_iter, th in
           enumerate([i * (2 * math.pi) / side for i in range(side)])]
    draw.polygon(loc, fill=f_c, outline=f_c)
    return previous_im


def get_location(shape_index, imshape, no_object_per_color):
    rotate_degree = np.random.randint(0, 360, no_object_per_color)
    if shape_index == 0 or shape_index == 1 or shape_index == 5 or shape_index == 6:
        loc_x = np.random.randint(0, imshape[1] - 60, no_object_per_color)
        loc_y = np.random.randint(0, imshape[0] - 60, no_object_per_color)
        width = [np.random.randint(0, imshape[1] - loc_x[i], 1) for i in range(no_object_per_color)]
        height = [np.random.randint(0, imshape[0] - loc_y[i], 1) for i in range(no_object_per_color)]
        return [loc_x, loc_y, width, height], rotate_degree
    elif shape_index == 2 or shape_index == 3:
        x_scale = np.random.randint(0, 100, no_object_per_color)
        y_scale = np.random.randint(0, 100, no_object_per_color)
        return [x_scale, y_scale], rotate_degree
    elif shape_index == 4:
        loc_0 = np.reshape(np.random.randint(0, 65, 2 * no_object_per_color), [no_object_per_color, 2])
        loc_1 = np.reshape(np.random.randint(70, 120, 2 * no_object_per_color), [no_object_per_color, 2])
        loc_2 = np.concatenate([loc_0[:, 1:], loc_1[:, 1:]], axis=1)

        loc_n = [[loc_0[i], loc_1[i], loc_2[i]] for i in range(no_object_per_color)]

        return loc_n, rotate_degree


def get_loc_for_even_number_object(image_size, num_obj_x, num_obj_y):
    x_split = np.linspace(0, image_size[1], num_obj_x + 1)
    y_split = np.linspace(0, image_size[0], num_obj_y + 1)
    x_y_g = [x_split, y_split]
    x_y_split = [[], []]
    for q in range(1):
        direc2change = x_y_g[q]
        rest_index = np.delete(np.arange(2), q)[0]
        direc2same = x_y_g[rest_index]
        for i in range(len(direc2change) - 1):
            x_y_split[q].append([[direc2change[i], direc2change[i + 1]] for _ in range(len(direc2same) - 1)])
            x_y_split[rest_index].append([[direc2same[j], direc2same[j + 1]] for j in range(len(direc2same) - 1)])
    for i in range(2):
        x_y_split[i] = [v for q in x_y_split[i] for v in q]
    return x_y_split


def get_loc_for_oddeneven_number_object(image_size, num_obj):
    x_split = np.linspace(0, image_size[1], num_obj + 1)
    y_split = np.linspace(0, image_size[0], num_obj + 1)
    x_y_g = [x_split, y_split]
    x_y_split = [[], []]
    for q in range(2):
        direc2change = x_y_g[q]
        direc2same = [0, image_size[0]]
        x_y_split[q].append([[direc2change[i], direc2change[i + 1]] for i in range(len(x_split) - 1)])
        x_y_split[np.delete(np.arange(2), q)[0]].append([direc2same for _ in range(len(x_split) - 1)])
    for i in range(2):
        x_y_split[i] = [v for q in x_y_split[i] for v in q]
    return x_y_split


def get_split_loc_for_multiple_object(imshape, no_object_per_color):
    x_y_split = get_loc_for_oddeneven_number_object(imshape, no_object_per_color)
    if no_object_per_color % 2 == 0 and no_object_per_color != 2:
        even_x_y_split_1 = get_loc_for_even_number_object(imshape, 2, no_object_per_color // 2)
        if no_object_per_color // 2 != 2:
            even_x_y_split_2 = get_loc_for_even_number_object(imshape, no_object_per_color // 2, 2)
        for i in range(2):
            [x_y_split[i].append(v) for v in even_x_y_split_1[i]]
            if no_object_per_color // 2 != 2:
                [x_y_split[i].append(v) for v in even_x_y_split_2[i]]

    return x_y_split


def get_location_multiple_object(shape_index, imshape, no_object_per_color, no_object_per_image):
    x_y_split = get_split_loc_for_multiple_object(imshape, no_object_per_image)
    if shape_index == 0 or shape_index == 1 or shape_index == 5 or shape_index == 6:
        loc_group = []
        for i in range(len(x_y_split[0])):
            x_split, y_split = x_y_split[0][i], x_y_split[1][i]
            loc_x = np.random.randint(x_split[0], x_split[1] - 10, no_object_per_color)
            loc_y = np.random.randint(y_split[0], y_split[1] - 10, no_object_per_color)
            width = [np.random.randint(10, x_split[1] - loc_x[i], 1)[0] for i in range(no_object_per_color)]
            height = [np.random.randint(10, y_split[1] - loc_y[i], 1)[0] for i in range(no_object_per_color)]
            if shape_index == 0 or shape_index == 5:
                width_height = np.min(np.concatenate([np.expand_dims(width, axis=1),
                                                      np.expand_dims(height, axis=1)], axis=1), axis=1)
                loc_group.append([loc_x, loc_y, width_height + loc_x, width_height + loc_y])
            else:
                loc_group.append([loc_x, loc_y, width + loc_x, height + loc_y])
        loc_group = np.transpose(loc_group, (2, 0, 1))
        return loc_group
    elif shape_index == 2 or shape_index == 3:
        loc_group = []
        for i in range(len(x_y_split[0])):
            x_split, y_split = x_y_split[0][i], x_y_split[1][i]
            x_scale = np.random.randint(x_split[0], x_split[1], no_object_per_color)
            y_scale = np.random.randint(y_split[0], y_split[1], no_object_per_color)
            loc_group.append([x_scale, y_scale])
        return np.transpose(loc_group, (2, 0, 1))
    elif shape_index == 4:
        loc_group = []
        for i in range(len(x_y_split[0])):
            x_split, y_split = x_y_split[0][i], x_y_split[1][i]
            loc_0 = np.zeros([no_object_per_color, 3])
            x_dist = (x_split[1] - x_split[0]) / 3
            for j in range(3):
                x_init = x_split[0] + j * x_dist
                x_end = x_split[1] - (2 - j) * x_dist
                _loc = np.random.randint(x_init, x_end, no_object_per_color)
                loc_0[:, j] = _loc
            y_dist = (y_split[1] - y_split[0]) / 3
            loc_1 = np.zeros([no_object_per_color, 3])
            for j in range(3):
                y_init = y_split[0] + j * y_dist
                y_end = y_split[1] - (2 - j) * y_dist
                _loc = np.random.randint(y_init, y_end, no_object_per_color)
                loc_1[:, j] = _loc
            loc_n = [[[loc_0[j, 0], loc_1[j, 0]],
                      [loc_0[j, 1], loc_1[j, 1]], [loc_0[j, 2], loc_1[j, 2]]] for j in range(no_object_per_color)]
            loc_group.append(loc_n)
        loc_group = np.transpose(loc_group, (1, 0, 2, 3))
        return loc_group


def draw_shape_with_multiple_objects_given_color(image_shape, shape_index, num_object_per_color, num_object_per_im,
                                                 color_info,
                                                 side, tds_dir, save=False):
    b_c, bg_index, color_g = color_info
    loc_group = get_location_multiple_object(shape_index, image_shape, num_object_per_color,
                                             num_object_per_im)
    im_g = []
    for i in range(num_object_per_color):
        loc = loc_group[i]
        for q in range(len(loc) // num_object_per_im):
            f_c = np.array(color_g)[np.random.choice(np.delete(np.arange(len(color_g)), bg_index), num_object_per_im)]
            f_c = tuple([tuple(v) for v in f_c])
            for j in range(num_object_per_im):
                _loc_index = q * num_object_per_im + j
                if j == 0:
                    if shape_index in [0, 1]:
                        im = draw_rectangle(image_shape, b_c, f_c[j],
                                            tuple(loc[_loc_index]), previous_im=[])
                    elif shape_index in [2, 3]:
                        im = draw_polygon(image_shape, b_c, f_c[j], side, loc[_loc_index][0],
                                          loc[_loc_index][1], previous_im=[])
                    elif shape_index in [4]:
                        im = draw_triangle(image_shape, b_c, f_c[j],
                                           tuple([tuple(v) for v in loc[_loc_index]]))
                    elif shape_index in [5, 6]:
                        im = draw_circle(image_shape, b_c, f_c[j], tuple(loc[_loc_index]))
                else:
                    if shape_index in [0, 1]:
                        im = draw_rectangle(image_shape, b_c, f_c[j],
                                            tuple(loc[_loc_index]), previous_im=im)
                    elif shape_index in [2, 3]:
                        im = draw_polygon(image_shape, b_c, f_c[j], side, loc[_loc_index][0],
                                          loc[_loc_index][1], previous_im=im)
                    elif shape_index in [4]:
                        im = draw_triangle(image_shape, b_c, f_c[j],
                                           tuple([tuple(v) for v in loc[_loc_index]]), im)
                    elif shape_index in [5, 6]:
                        im = draw_circle(image_shape, b_c, f_c[j], tuple(loc[_loc_index]), previous_im=im)

            if save:
                _obj_index = i * (len(loc) // num_object_per_im) + q
                im.save(tds_dir + "/shape_%d_bgColor_%d_fgColor_%d_NoObject_%d_%d.png" % (shape_index,
                                                                                          bg_index,
                                                                                          100,
                                                                                          num_object_per_im,
                                                                                          _obj_index))
            else:
                im_g.append(im)
    if not save:
        return im_g


def draw_shape_with_multiple_object(color_use, num_object_per_color, num_object_per_im, shape_index, imshape, tds_dir,
                                    save=False):
    """Args:
        color_use: color value
        num_object_per_color: int
        num_object_per_im: int
        shape_index: int
        imshape: tuple (imh, imw)
        tds_dir: str
        save: bool variable
    """
    side = [5 if shape_index == 2 else 6][0]
    for i, s_bg in enumerate(color_use):
        draw_shape_with_multiple_objects_given_color(imshape, shape_index, num_object_per_color, num_object_per_im,
                                                     [s_bg, i, color_use],
                                                     side, tds_dir, save=save)


def draw_shape_wt_one_object(color_use, no_object_per_color, shape_index, imshape, tds_dir):
    """Args:
        color_use: [bg_color, fg_color]
        no_object_per_color: int, the number of shapes that are with the same color
        shape_index: int, which shape-function to run
        imshape: (imh, imw)
        tds_dir: str, the directory to save the images
    Ops:
        shape_index:
        0: square
        1: rectangle
        2: 5 sides polygan
        3: 6 sides polygan
        4. triangle
        5. circle
        6. eclipse
    """
    num_c = len(color_use)
    rotate_degree = np.random.randint(0, 360, no_object_per_color)
    for i, s_bg in enumerate(color_use):
        fg_index = np.delete(np.arange(num_c), i)
        fg_color = [tuple(v) for v in np.array(color_use)[fg_index]]
        for j, s_fg in enumerate(fg_color):
            if shape_index in [0, 1, 5, 6]:
                [loc_x, loc_y, width, height], rotate_degree = get_location(shape_index, imshape, no_object_per_color)
            elif shape_index in [2, 3]:
                [x_scale, y_scale], rotate_degree = get_location(shape_index, imshape, no_object_per_color)
            elif shape_index == 4:
                loc, rotate_degree = get_location(shape_index, imshape, no_object_per_color)
            for k in range(no_object_per_color):
                if shape_index == 0:
                    im = draw_rectangle(imshape, s_bg, s_fg,
                                        tuple([loc_x[k], loc_y[k], loc_x[k] + width[k], loc_y[k] + width[k]]))
                elif shape_index == 1:
                    im = draw_rectangle(imshape, s_bg, s_fg,
                                        tuple([loc_x[k], loc_y[k], loc_x[k] + width[k], loc_y[k] + height[k]]))
                elif shape_index == 2:
                    im = draw_polygon(imshape, s_bg, s_fg, 5, x_scale[k], y_scale[k])
                elif shape_index == 3:
                    im = draw_polygon(imshape, s_bg, s_fg, 6, x_scale[k], y_scale[k])
                elif shape_index == 4:
                    im = draw_triangle(imshape, s_bg, s_fg, tuple([tuple(v) for v in loc[k]]))
                elif shape_index == 5:
                    im = draw_circle(imshape, s_bg, s_fg,
                                     tuple([loc_x[k], loc_y[k], loc_x[k] + width[k], loc_y[k] + width[k]]))
                elif shape_index == 6:
                    im = draw_circle(imshape, s_bg, s_fg,
                                     tuple([loc_x[k], loc_y[k], loc_x[k] + width[k], loc_y[k] + height[k]]))
                if shape_index not in [4, 5]:
                    im = im.rotate(rotate_degree[k], fillcolor=s_bg)
                
                im.save(tds_dir + "/shape_%d_bgColor_%d_fgColor_%d_NoObject_1_%d.png" % (shape_index,
                                                                                         i, fg_index[j], k))
                
                
def draw_shape_black_and_white(shape_index, imshape, num_object, tds_dir):
    rotate_degree = np.random.randint(0, 360, num_object)
    s_bg = (0, 0, 0)
    if shape_index in [0, 1, 5, 6]:
        [loc_x, loc_y, width, height], rotate_degree = get_location(shape_index, imshape, num_object)
    elif shape_index in [2, 3]:
        [x_scale, y_scale], rotate_degree = get_location(shape_index, imshape, num_object)
    elif shape_index == 4:
        loc, rotate_degree = get_location(shape_index, imshape, num_object)
    for k in range(num_object):
        s_fg = tuple(np.repeat(int(np.random.uniform(0.2, 1, 1)[0] * 225), 3))
        if shape_index == 0:
            im = draw_rectangle(imshape, s_bg, s_fg,
                                tuple([loc_x[k], loc_y[k], loc_x[k] + width[k], loc_y[k] + width[k]]))
        elif shape_index == 1:
            im = draw_rectangle(imshape, s_bg, s_fg,
                                tuple([loc_x[k], loc_y[k], loc_x[k] + width[k], loc_y[k] + height[k]]))
        elif shape_index == 2:
            im = draw_polygon(imshape, s_bg, s_fg, 5, x_scale[k], y_scale[k])
        elif shape_index == 3:
            im = draw_polygon(imshape, s_bg, s_fg, 6, x_scale[k], y_scale[k])
        elif shape_index == 4:
            im = draw_triangle(imshape, s_bg, s_fg, tuple([tuple(v) for v in loc[k]]))
        elif shape_index == 5:
            im = draw_circle(imshape, s_bg, s_fg,
                                tuple([loc_x[k], loc_y[k], loc_x[k] + width[k], loc_y[k] + width[k]]))
        elif shape_index == 6:
            im = draw_circle(imshape, s_bg, s_fg,
                                tuple([loc_x[k], loc_y[k], loc_x[k] + width[k], loc_y[k] + height[k]]))
        if shape_index not in [4, 5]:
            im = im.rotate(rotate_degree[k], fillcolor=s_bg)
        im = im.resize((28, 28))
        im.save(tds_dir + "/shape_%02d_%05d.png" % (shape_index, k))

    
def draw_shape_with_same_size_as_mnist():
    num_im_per_shape = [50000 / 6, 10000/ 6, 10000 / 6]
    folder = ["train", "val", "test"]
    tds_dir = "toy_shape/"
    for s_folder, s_num_im_per_shape in zip(folder, num_im_per_shape):
        num_im_tot = 0.0
        dir_use = tds_dir + "%s/" % s_folder 
        if not os.path.exists(dir_use):
            os.makedirs(dir_use)
        for shape_index in [0, 1, 2, 3, 5, 6]:
            draw_shape_black_and_white(shape_index, (128, 128), int(s_num_im_per_shape), dir_use)
            num_im_tot += int(s_num_im_per_shape)
        print("Finishing drawing shape for", s_folder, num_im_tot)
    

        
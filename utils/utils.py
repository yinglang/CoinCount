import os

"""
1. some gnerally tool function
"""
def mkdir_if_not_exist(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
  

"""
2. dataset transform
"""
from PIL import Image
import os
import numpy as np
def resize_imageset(image_root, out_dir, resize, resample=Image.BOX):
    for imgname in os.listdir(image_root):
        imgpath = image_root + "/" + imgname
        img = Image.open(imgpath)
        img = img.resize(resize, resample)
        img.save(out_dir + "/" + imgname)

def cal_mean(path_root):
    mean = np.array([0., 0., 0.])
    for imgpath in os.listdir(path_root):
        img = Image.open(os.path.join(path_root, imgpath))
        img = np.array(img)
        mean += np.mean(img, axis=(0, 1))
    mean /= len(os.listdir(path_root))
    return mean

def cal_mean_std(path_root):
    imgs = None
    for imgpath in os.listdir(path_root):
        img = Image.open(os.path.join(path_root, imgpath))
        img = np.array(img).reshape((-1, 3))
        if imgs is None:
            imgs = img
        else:
            imgs = np.concatenate((imgs, img), axis=0)
    return np.mean(imgs, axis=0), np.std(imgs, axis=0)
        
"""
3. data prepared
"""
import os
import shutil
import xml.dom.minidom as d
from PIL import Image

__label_dict = {}
def get_info_from_annotaions_VOC(annopath, normalize=True):
    """
    args:
        annopath: VOC format annotaion file path
        normalzie: whether boxes is normalzie to [0, 1] (/w, /h)
    
    return:
        boxes: ground truth boxes for detection.
        labels: ground truth boxes's class id, same length with boxes.
        w: image width recored in annoaton file.
        h: image height recored in annoaton file.
    """
    def get_label_id(name):
        if not __label_dict.has_key(name):
            __label_dict[name] = len(__label_dict)
        return __label_dict[name]
    
    dom = d.parse(annopath)
    root = dom.documentElement
    
    size = root.getElementsByTagName("size")[0]
    w = float(size.getElementsByTagName('width')[0].childNodes[0].data)
    h = float(size.getElementsByTagName('height')[0].childNodes[0].data)
    
    boxes = []
    labels = []
    for o in root.getElementsByTagName("object"):
        l = get_label_id(o.getElementsByTagName('name')[0].childNodes[0].data)
        bd = o.getElementsByTagName('bndbox')[0];
        x0 = bd.getElementsByTagName('xmin')[0].childNodes[0].data
        y0 = bd.getElementsByTagName('ymin')[0].childNodes[0].data
        x1 = bd.getElementsByTagName('xmax')[0].childNodes[0].data
        y1 = bd.getElementsByTagName('ymax')[0].childNodes[0].data
        
        if normalize:
            box = [float(x0)/w, float(y0)/ h, float(x1)/w, float(y1)/h]
        else:
            box = [int(x0), int(y0), int(x1), int(y1)]
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
            labels.append(l)
        else:
            print str(box) + " is not valid box in " + annopath + ", just ignore."
    info = {"boxes": boxes, "labels": labels, "w": w, "h": h}
    return info

def get_info_from_annotaions_SDL(annopath, imgpath, normalize=True):
    """
    args:
        annopath: VOC format annotaion file path
        imgpath: the image path to annotaion file.
        normalzie: whether boxes is normalzie to [0, 1] (/w, /h)
    
    return:
        boxes: ground truth boxes for detection.
        labels: ground truth boxes's class id.
        w: image width recored in annoaton file.
        h: image height recored in annoaton file.
    """
    def get_image_wh(imgpath):
        img = Image.open(imgpath)
        width = img.width
        height = img.height
        return width, height
    
    def split_label(label):
        label = label.strip(' ').strip('\r\n').strip('\n')
        if len(label.split('\t')) >= 9: return label.split('\t')  # not seperate with '\t'
        label = label.split(' ')
        new_label = []
        for l in label:
            if len(l) > 0:
                new_label.append(l)
        return new_label
    
    w, h = get_image_wh(imgpath)
    
    windows = open(annopath).readlines()
    boxes = []
    labels = []
    angles = []
    for label in windows:
        label = split_label(label)
        label = map(lambda x: float(x), label[:-1])
        x1, y1, x2, y2, x3,y3, x4, y4 = label[:8]
        xmin = min([x1, x2, x3, x4])
        ymin = min([y1, y2, y3, y4])
        xmax = max([x1, x2, x3, x4])
        ymax = max([y1, y2, y3, y4])
        if normalize:
            box = (xmin/w, ymin/h, xmax/w, ymax/h)
        else:
            box = (xmin, ymin, xmax, ymax)
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
            labels.append(label[-1])
            angles.append(label[-2])
        else:
            print str(box) + " is not valid box in " + annopath + ", just ignore."
    info = {"boxes": boxes, "labels": labels, "w": w, "h": h, 'angles': angles}
    return info

def get_info_from_annotaions(annopath, fmt, normalize=True, **kwargs):
    if fmt.lower() == "voc":
        return get_info_from_annotaions_VOC(annopath, normalize)
    elif fmt.lower() == "sdl":
        return get_info_from_annotaions_SDL(annopath, kwargs['imgpath'], normalize)
    else:
        raise ValueError("annotation format is not support, annotaion fomrat must be on of [VOC, SDL]")

def list_image_det(lst_file, annotations_root, out_lst_file=None, fmt="VOC", path_root=None, resize=None, resize_out_dir=None):
    """
    lst_file: .lst file generate by im2rec.py
    out_lst_file: output .lst file after modyfied
    annotations_root: annaotaion file's root dir
    fmt: could be 'SDL' or 'VOC'
    path_root: if fmt is 'SDL', then path_root need to spcified, it is direcroty prefix of image path in .lst file
    resize: (w, h) pair. use to set (w, h) in lst file, will not really do resize if resize_out_dir is not specified.
    resie_out_dir: will do resize to imageset, and path_root must be specified.
    """
    # some check
    fmt = fmt.lower()
    if fmt == 'voc':
        ext = ".xml"
    elif fmt == 'sdl':
        if path_root  is None:
            raise ValueError("when data is SDL formt, 'path_root' must specified, it will use to open image.")
        ext = '.txt'
    else:
        raise ValueError("annotation format is not support, 'fmt' must be on of [VOC, SDL]")
    if (resize_out_dir is not None) and path_root is None:
        raise ValueError("path_root must be specified to the dir of images when resize_out_dir has been specified.")
    
    def boxes_to_str(boxes, labels):
        s = ""
        for i, box in enumerate(boxes):
            box = [str(it) for it in box]
            s += str(labels[i]) + "\t" + "\t".join(box) + "\t"
        return s[:-1]
    
    lst_f = open(lst_file)
    if out_lst_file is None:
        o_lst_f = open("tmp", 'w')
    else:
        o_lst_f = open(out_lst_file, 'w')
        
    for line in lst_f.readlines():
        items = line.split('\t')
        imgname = items[-1].split("/")[-1]
        imgname = imgname.split(".")[0]
        
        new_line = items[0] + "\t"
        
        annopath = os.path.join(annotations_root, imgname+ext)
        if fmt == 'voc':
            info = get_info_from_annotaions(annopath, fmt)
        if fmt == 'sdl':
            info = get_info_from_annotaions(annopath, fmt, 
                                                           imgpath=path_root+"/"+items[-1].strip('\n'))
        boxes, labels, w, h = info['boxes'], info['labels'], info['w'], info['h']
        new_line += "4\t" + str(5) + "\t" 
        if resize is not None:
            new_line += str(resize[0]) + '\t' + str(resize[1]) + '\t'
        else:
            new_line += str(w) + '\t' + str(h) + '\t'
        new_line += boxes_to_str(boxes, labels) + "\t"
        
        new_line += items[-1]
        o_lst_f.write(new_line)
    lst_f.close()
    o_lst_f.close()
    
    if out_lst_file is None:
        shutil.move('tmp', lst_file)
        
    if resize_out_dir is not None:
        resize_imageset(path_root, resize_out_dir, resize)
        
    return __label_dict


"""
4. data visualize
""" 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
def show_image_SDL_annotation(imgpath, annopath=None, color=(0, 1, 0)):
    """
    show sigle image with sdl format annotation
    """
    def draw_rect(x0, y0, x1, y1, color):
        plt.plot([x0, x1],[y0, y0], color=color)
        plt.plot([x1, x1],[y0, y1], color=color)
        plt.plot([x0, x1],[y1, y1], color=color)
        plt.plot([x0, x0],[y0, y1], color=color)
    fig = plt.figure(figsize=(16, 16), dpi=72)
    image = Image.open(imgpath)
    plt.imshow(np.asarray(image))
    if annopath is not None:
        info = get_info_from_annotaions_SDL(annopath, imgpath, False)
        for box in info["boxes"]:
            draw_rect(box[0], box[1], box[2], box[3], color)
    plt.show()

def try_asnumpy(data):
    try:
        data = data.asnumpy() # if is <class 'mxnet.ndarray.ndarray.NDArray'>
    except BaseException:
        pass
    return data

def box_to_rect(box, color, linewidth=1):
    """convert an anchor box to a matplotlib rectangle"""
    return plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                  fill=False, edgecolor=color, linewidth=linewidth)

def show_images(images, labels=None, rgb_mean=np.array([0, 0, 0]), 
                MN=None, color=(0, 1, 0), linewidth=1, figsize=(8, 4), show_text=False, fontsize=5):
    """
    advise to set dpi to 120
        import matplotlib as mpl
        mpl.rcParams['figure.dpi'] = 120
    
    images: numpy images type, shape is (n, 3, h, w), or (n, 2, h, w)
    labels: boxes, shape is (n, m, 5), m is number of box, 5 means every box is [label_id, xmin, ymin, xmax, ymax]
    rgb_mean: if images has sub rgb_mean, shuold specified.
    MN: is subplot's row and col, defalut is (-1, 5), -1 mean row is adaptive, and col is 5
    """
    images = try_asnumpy(images)
    labels = try_asnumpy(labels)
    
    if MN is None:
        M, N = (images.shape[0] + 4) / 5, 5
    else:
        M, N = MN
    _, figs = plt.subplots(M, N, figsize=figsize)
    
    images = images.transpose((0, 2, 3, 1)) + rgb_mean
    h, w = images.shape[1], images.shape[2]
    for i in range(M):
        for j in range(N):
            if N * i + j < images.shape[0]:
                image = (images[N * i + j] / 255).clip(0, 1)
                figs[i][j].imshow(image)
                if labels is not None:
                    label = labels[N * i + j]
                    for l in label:
                        if l[0] < 0: continue
                        l[1], l[2], l[3], l[4] = l[1] * w, l[2] * h, l[3] * w, l[4] * h
                        rect = box_to_rect(l[1:5], color, linewidth)
                        figs[i][j].add_patch(rect)
                        if show_text:
                            figs[i][j].text(l[1], l[2], str(int(l[0])), 
                                            bbox=dict(facecolor=(1, 1, 1), alpha=0.5), fontsize=fontsize, color=(0, 0, 0))

                    figs[i][j].axes.get_xaxis().set_visible(False)
                    figs[i][j].axes.get_yaxis().set_visible(False)
            else:
                figs[i][j].set_visible(False)
    plt.show()

def show_9_images(images, labels=None, rgb_mean=np.array([0, 0, 0]), color=(0, 1, 0), linewidth=1, **kwargs):
    """
    invoke show_images with MN=(3, 3)
    """
    show_images(images, labels, rgb_mean, (3, 3), color, linewidth, figsize=(6, 6), **kwargs)
    
    
def show_det_result(im, out, threshold=0.5, class_names=None, colors = ['blue', 'green', 'red', 'black', 'magenta']):
    """
    im: image data, numpy.array or ndarray
    out: detection result, numpy.array or ndarray
    theshold: score threshold
    class_name: class or labels name
    """
    im = try_asnumpy(im)
    out = try_asnumpy(out)
    
    plt.imshow(im)
    for row in out:
        class_id, score = int(row[0]), row[1]
        if class_id < 0 or score < threshold:  # class_id < 0 is background rect
            continue
        color = colors[class_id%len(colors)]
        box = row[2:6] * np.array([im.shape[0],im.shape[1]]*2)
        rect = box_to_rect(box, color, 2)
        plt.gca().add_patch(rect)

        text = class_names[class_id] if class_names else "class " + str(class_id)
        plt.gca().text(box[0], box[1],
                       '{:s} {:.2f}'.format(text, score),
                       bbox=dict(facecolor=color, alpha=0.5),
                       fontsize=10, color='white')
    plt.show()

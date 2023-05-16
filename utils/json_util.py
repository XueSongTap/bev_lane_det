import json
import os
from tqdm import tqdm


def load_from_list(root, file_list, parser, output_path=False):
    ret = []
    for fn in file_list:
        if output_path:
            print(os.path.join(root, fn), end="\t")
        with open(os.path.join(root, fn)) as fp:
            jo = json.load(fp)
            ret.append(parser(jo))
    return ret


def load_from_list_with_fn(root, file_list, parser, output_path=False, progress=False, full_path=False):
    ret = []
    if progress:
        file_list = tqdm(file_list)

    for fn in file_list:
        if output_path:
            print(os.path.join(root, fn), end="\t")
        with open(os.path.join(root, fn)) as fp:
            jo = json.load(fp)
            if full_path:
                parser_fn = os.path.join(root, fn)
            else:
                parser_fn = fn
            ret.append(parser(jo, parser_fn))
    return ret


def load_from_list_with_fn_ray(root, file_list, parser, output_path=False):
    ret = []
    for fn in file_list:
        if output_path:
            print(os.path.join(root, fn), end="\t")
        with open(os.path.join(root, fn)) as fp:
            jo = json.load(fp)
            ret.append(parser.remote(jo, fn))
    return ret

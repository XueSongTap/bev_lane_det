import cv2


def cv2_load(path, cvt=True):
    image = cv2.imread(path)
    if image is None:
        return None
    if cvt:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
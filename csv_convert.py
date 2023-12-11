import pandas as pd
from PIL import Image
import numpy as np


def image_to_csv(path):
    image = Image.open(path).convert('L')  # открываем картинку и переводим в черно-белый формат
    image_array = np.array(image)  # преобразуем картинку в массив numpy

    df = pd.DataFrame(image_array)

    df = 255 - df

    path = "image.csv"
    df.to_csv("image.csv", index=False, header=False)
    return path

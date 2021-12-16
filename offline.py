
from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model

if __name__ == '__main__':
    fe = FeatureExtractor(load_model('./static/model/IR.h5', compile=False))

    for img_path in sorted(Path("./static/image_db").glob("*.jpg")):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path("./static/feature") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)
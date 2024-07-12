# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from datetime import datetime
import io
import pstats
import subprocess
import time
import cv2
import os
import tempfile
from basicsr.archs.rrdbnet_arch import RRDBNet
from cog import BasePredictor, Input, Path
import cProfile

from gfpgan import GFPGANer
from mod_model import RealESRGANerCompile
import pandas as pd

WEIGHTS_URL = "https://weights.replicate.delivery/default/official-models/tencent/real-esrgan/real-esrgan-models.tar"
EXTRA_URL = "https://weights.replicate.delivery/default/official-models/tencent/real-esrgan/esrgan-extra-models.tar"

MODEL_FOLDER = "/src/weights/esrgan/"
GFPGAN_FOLDER = "/src/gfpgan/weights/"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self):
        if not os.path.exists(MODEL_FOLDER):
            download_weights(WEIGHTS_URL, MODEL_FOLDER)
        if not os.path.exists(GFPGAN_FOLDER):
            download_weights(EXTRA_URL, GFPGAN_FOLDER)

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
        self.upsampler = RealESRGANerCompile(
            scale=netscale,
            model_path=os.path.join(MODEL_FOLDER, "RealESRGAN_x4plus.pth"),
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
        )
        self.face_enhancer = GFPGANer(
            model_path=os.path.join(MODEL_FOLDER, "GFPGANv1.3.pth"),
            upscale=4,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=self.upsampler,
        )

    def predict(
        self,
        image: Path = Input(description="Input image"),
        scale: float = Input(
            description="Factor to scale image by", ge=0, le=10, default=4
        ),
        face_enhance: bool = Input(
            description="Run GFPGAN face enhancement along with upscaling",
            default=False,
        ),
        profile: bool = Input(
            description="Run profiler",
            default=False
        ),
        expt: str = Input(
            description="Where do you want to store your profiles?",
            default=None
        )
    ) -> Path:
        pr = cProfile.Profile()
        pr.enable()
        img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)

        if face_enhance:
            print("running with face enhancement")
            self.face_enhancer.upscale = scale
            _, _, output = self.face_enhancer.enhance(
                img, has_aligned=False, only_center_face=False, paste_back=True
            )
        else:
            print("running without face enhancement!!")
            print(img.shape)
            output, _ = self.upsampler.enhance(img, outscale=scale)
            print('urpsam')
        save_path = "output.png"
        cv2.imwrite(save_path, output)
        pr.disable()
        print('done profiling')
        urpscale = 4 if face_enhance else scale
        print(image)
        n_faces = int(str(image).split("gan")[1][0]) if face_enhance else -1
        image_name = str(image).split('/')[-1]
        if profile:
            if not expt:
                expt = int(time.time())
            store_profile(pr, urpscale, n_faces, img.shape[0] * img.shape[1], image_name, expt)
        return Path(save_path)


def store_profile(pr, scale, n_faces, n_pixels, image_name, expt):
    profile_fpath = os.path.join('/src/profiles', expt)
    if not os.path.exists(profile_fpath):
        os.makedirs(profile_fpath)
    s = io.StringIO()
    print('borm')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    prof_path = os.path.join(profile_fpath, 'prof')
    if not os.path.exists(prof_path):
        os.makedirs(prof_path)

    pr.dump_stats(os.path.join(prof_path, f"{image_name}_{scale}_{timestamp}.prof"))
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print('potato')
    # Get the string value and split it into lines
    lines = s.getvalue().split('\n')

    # Parse the relevant lines into a list of dictionaries
    data = []
    for line in lines[5:]:  # Skip header lines
        if line.strip():
            items = line.split()
            if len(items) == 6:
                data.append({
                    'ncalls': items[0],
                    'tottime': float(items[1]),
                    'percall_tot': float(items[2]),
                    'cumtime': float(items[3]),
                    'percall_cum': float(items[4]),
                    'filename:lineno(function)': items[5],
                    'scale': scale,
                    'n_faces': n_faces,
                    'n_pixels': n_pixels
                })
    # Create a DataFrame
    df = pd.DataFrame(data)
    print(df.shape)
    # Generate a unique filename with timestamp
    df_path = os.path.join(profile_fpath, 'df')
    if not os.path.exists(df_path):
        os.makedirs(df_path)
    filename = os.path.join(df_path, f"{image_name}_{scale}_{timestamp}.csv")

    # Save to CSV
    df.to_csv(filename, index=False)

    print(f"Profile results saved to {filename}")

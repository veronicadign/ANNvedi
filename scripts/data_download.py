import gdown
import os

folder_id = "1kDMoujUUYc0DbAMw4WGRtlXcruEVFCEd"
output_dir = "dataset"  # repo convention: singular dataset/ dir

os.makedirs(output_dir, exist_ok=True)

gdown.download_folder(
    id=folder_id,
    output=output_dir,
    quiet=False,
    use_cookies=False
)

hdf5_files = [f for f in os.listdir(output_dir) if f.endswith(".hdf5")]
print("Datasets trovati:", hdf5_files)
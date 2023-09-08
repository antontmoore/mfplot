import argparse
from pathlib import Path
import shutil
import warnings
import pickle
from google.cloud import storage
import sys
from distutils.util import strtobool

sys.path.append('../')
from data_model.waymo_convertor import WaymoConvertor

BF_PATH_PREFIX = "uncompressed/tf_example/"  # path prefix for bucket from
DIRS = ['training', 'validation', 'testing']


def convert_bucket(bucket_from_name,
                   bucket_to_name,
                   dir_for_download,
                   dir_for_store,
                   skip_existing=True,
                   max_parts=0,
                   max_scenes=0):

    wc = WaymoConvertor()

    storage_client = storage.Client(project="grumpynet")
    bucket_from = storage.Bucket(storage_client, bucket_from_name)
    bucket_to = storage.Bucket(storage_client, bucket_to_name)

    for tvt_name in DIRS:
        print(f"\n📂 Converting dir: \t{tvt_name}")
        for parts_counter, blob in enumerate(bucket_from.list_blobs(prefix=BF_PATH_PREFIX + tvt_name)):
            filename_path_splitted = blob.name.split('/')
            dirname, filename = filename_path_splitted[-2:]
            part_name = "part" + filename.split("-")[-3]



            # paths for download and saving
            p = Path() / dir_for_download / dirname
            path_download_dir = p.resolve()
            p = Path() / dir_for_store / tvt_name / part_name
            path_dir_for_saving = p.resolve()

            # if we already have next dir in destination bucket, this folder is already converted
            if skip_existing and exist_next_dir(bucket_to, path_dir_for_saving, part_name):
                print(f"  ⏭️ Skipping the part \t\t{filename}")
                continue

            # create dirs if not created
            make_dir_with_parents(path_download_dir)
            make_dir_with_parents(path_dir_for_saving)

            path_download_file = path_download_dir / filename
            if path_download_file.exists():
                print(f"  💾 File already exists!\t{filename}")
            else:
                print(f"  🌍 Downloading file: \t\t{filename}")
                blob.download_to_filename(path_download_file)

            print(f"  🚗 Converting file: \t\t{filename}")

            scene_counter = 0
            for scene, counter, total in wc.read_and_convert_dataset_part(path_download_file):
                print(f"    {scene.scene_id}  ->  {counter} of {total}")

                path2save = path_dir_for_saving / (scene.scene_id + ".pkl")
                path2save = str(path2save.absolute())

                with open(path2save, 'wb') as file_to_write:
                    pickle.dump(scene, file_to_write)
                    path2save_in_bucket = "waymo/" + "/".join(path2save.split('/')[-3:])
                    blob = bucket_to.blob(path2save_in_bucket)
                    blob.upload_from_filename(path2save)

                scene_counter += 1
                if max_scenes and scene_counter >= max_scenes:
                    print(f"  🚧 Processed maximum scenes of the part (max_parts = {max_scenes})\n")
                    break

            if not max_scenes:
                print(f"  🚙 Converted file: \t\t{filename}")

            # removing big files from file store
            path_download_file.unlink()
            shutil.rmtree(path_dir_for_saving)

            if max_parts and parts_counter + 1 >= max_parts:
                print(f"Processed maximum parts of the directory (max_parts = {max_parts})\n")
                break


def make_dir_with_parents(path):

    levels = len(str(path).split('/')) - 1
    for level in range(levels-1, -1, -1):
        if not path.parents[level].exists():
            path.parents[level].mkdir()

    if not path.exists():
        path.mkdir()

def exist_next_dir(bucket_to, dir_path, part_name):

    path2part_in_bucket = "waymo/" + "/".join(str(dir_path).split('/')[-2:])
    partnum = int(path2part_in_bucket[-5:])
    next_part_num = str(partnum + 1)
    next_part_num = "0" * (5 - len(next_part_num)) + next_part_num

    path2nextpart_in_bucket = path2part_in_bucket[:-5] + next_part_num
    try:
        first = next(bucket_to.list_blobs(prefix=path2nextpart_in_bucket))
    except StopIteration:
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converting parameters')
    parser.add_argument('--bucket_from_name', type=str, default='waymo_open_dataset_motion_v_1_2_0')
    parser.add_argument('--bucket_to_name', type=str, default='motion_prediction_data_v_0_1')
    parser.add_argument('--dir_for_download', type=str, default='../data/waymo')
    parser.add_argument('--dir_for_store', type=str, default='../data/waymo_converted')
    parser.add_argument('--skip_existing', type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--max_parts', type=int, default=0)
    parser.add_argument('--max_scenes', type=int, default=0)
    params = parser.parse_args()
    convert_bucket(
        bucket_from_name=params.bucket_from_name,
        bucket_to_name=params.bucket_to_name,
        dir_for_download=params.dir_for_download,
        dir_for_store=params.dir_for_store,
        skip_existing=params.skip_existing,
        max_parts=params.max_parts,
        max_scenes=params.max_scenes
    )

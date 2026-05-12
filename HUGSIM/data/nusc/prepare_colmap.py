import os
import numpy as np
from pathlib import Path
import argparse
import json
from colmap.colmap import COLMAPAuto, rotmat2qvec


def get_opts():
    parser = argparse.ArgumentParser("colmap prepare", description='prepare colamp image dataset')
    parser.add_argument('-i', '--in_path', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_opts()

    with open(os.path.join(args.in_path, 'meta_data.json'), 'r') as jf:
        meta_data = json.load(jf)

    path_prior = os.path.join(args.in_path, 'prior')
    path_rigid = os.path.join(args.in_path, 'cam_rigid_config.json')
    os.makedirs(path_prior, exist_ok=True)

    # points3D
    Path(os.path.join(path_prior, 'points3D.txt')).touch()

    with open(os.path.join(path_prior, 'cameras.txt'), 'w') as f:
        for idx, frame in enumerate(meta_data['frames'][:6]):
            intr = np.array(frame['intrinsics'])
            w, h = frame['width'], frame['height']
            intr4 = [intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]]
            intr4 = [str(i.item()) for i in intr4]
            str_intr = ' '.join(intr4)
            f.write(f"{idx+1} PINHOLE {w} {h} {str_intr}" + '\n')

    # images
    with open(os.path.join(path_prior, 'images.txt'), 'w') as f:
        for idx, frame in enumerate(meta_data['frames']):
            c2w = np.array(frame['camtoworld'])
            img_path = frame['rgb_path']

            rel_path = os.path.relpath(img_path, "./images")

            w2c = np.linalg.inv(c2w)
            q_w2c = [str(v.item()) for v in rotmat2qvec(w2c[:3, :3])]
            t_w2c = [str(v.item()) for v in w2c[:3, -1]]
            cam_id = idx % 6 + 1
            line = f"{idx+1} {' '.join(q_w2c)} {' '.join(t_w2c)} {cam_id} {rel_path}"
            f.write(line + '\n\n')

    auto = COLMAPAuto(args.in_path)

    auto.feature_extract()

    # DB에 실제 등록된 image_id/camera_id 기준으로 images.txt, cameras.txt 재작성
    import sqlite3
    dbconn = sqlite3.connect(auto.path_database)
    cur = dbconn.cursor()
    name2db = {name: (iid, cid) for iid, name, cid in cur.execute('SELECT image_id, name, camera_id FROM images')}
    camid2folder = {cid: name.split('/')[0] for iid, name, cid in cur.execute('SELECT image_id, name, camera_id FROM images WHERE name LIKE "%/00000.%"')}
    dbconn.close()

    folder2intr = {}
    for frame in meta_data['frames'][:6]:
        folder = frame['rgb_path'].split('/')[-2]
        intr = np.array(frame['intrinsics'])
        w, h = frame['width'], frame['height']
        folder2intr[folder] = (w, h, intr[0,0], intr[1,1], intr[0,2], intr[1,2])

    with open(os.path.join(path_prior, 'cameras.txt'), 'w') as f:
        for cid, folder in sorted(camid2folder.items()):
            w, h, fx, fy, cx, cy = folder2intr[folder]
            f.write(f"{cid} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")

    with open(os.path.join(path_prior, 'images.txt'), 'w') as f:
        for idx, frame in enumerate(meta_data['frames']):
            img_name = '/'.join(frame['rgb_path'].replace('./images/', '').split('/')[-2:])
            iid, cid = name2db[img_name]
            c2w = np.array(frame['camtoworld'])
            w2c = np.linalg.inv(c2w)
            q_w2c = [str(v.item()) for v in rotmat2qvec(w2c[:3, :3])]
            t_w2c = [str(v.item()) for v in w2c[:3, -1]]
            f.write(f"{iid} {' '.join(q_w2c)} {' '.join(t_w2c)} {cid} {img_name}\n\n")

    auto.sequential_matcher()
    auto.point_triangulator()
    os.system(f'cp -r {auto.path_tri} {auto.path_ba}')
    auto.rigid_ba(path_rigid)
    auto.point_triangulator_ba()
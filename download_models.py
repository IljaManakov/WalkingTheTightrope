import argparse
import os
from itertools import product
from collections import Sequence

from google_drive_downloader import GoogleDriveDownloader as gdd


file_ids = {
    0: {
        'pokemon': '1okRM6Lqu5XJL2sQFrmOO1KifW2K4d9x2',
        'celeba': '1fu24Hj60gnazF1Wja4IGSzY1beQRzpKe',
        'stl-10': '1ANwkW-qXgZBayF7zNiHgrnwao7h6PLjb'
    },
    1: {
        'pokemon': '1Io2tRnwZq7RsUTl-CKNSDSVtafobp1tI',
        'celeba': '1H8QJbuf0VbB4e6VgMOfcyqr6zNLTierq',
        'stl-10': '1aXhoEzcDseYPUMBH1B6lIbnE24lYOR3X'
    },
    2: {
        'pokemon': '1DBVqWUmWFVwEM_yHvEKR_pdsPKafGzEM',
        'celeba': '1fuC6axLn1OhiYQxE-AcbIDSn6FrsLdBu',
        'stl-10': '1qORgKVfsY_y-xFzzW1eYdw-7BbSclp5-'
    }
}


def download(dest, file_id):

    gdd.download_file_from_google_drive(file_id=file_id, dest_path=dest, unzip=True)
    os.remove(dest)


if __name__ == '__main__':

    # parse cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest', type=str, default='./models',
                        help="""directory where to download the dataset""")
    parser.add_argument('--datasets', type=str, default=['pokemon', 'celeba', 'stl-10'], nargs='*',
                        help="""models tained on which data?""")
    parser.add_argument('--seeds', type=int, default=[0, 1, 2], nargs='*',
                        help="""models initialized from which seeds?""")

    args = parser.parse_args()
    datasets = args.datasets if isinstance(args.datasets, Sequence) else [args.datasets]
    seeds = args.seeds if isinstance(args.seeds, Sequence) else [args.seeds]
    dest = args.dest

    # download models
    for dataset, seed in product(datasets, seeds):
        filename = os.path.join(dest, f'{dataset}-{seed}.zip')
        id = file_ids[seed][dataset]
        download(filename, id)
    print('all done!')

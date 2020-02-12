import argparse
import os

from google_drive_downloader import GoogleDriveDownloader as gdd


file_ids = {
    'pokemon': '1PLWW05OeHCWAR6o91PI8OyCLbXpOYSrZ',
    'celeba': '1rECVTmP_JDILLu-FiT80LCS2IuPFqKqt',
    'stl-10': '1pdDpHVqnVF-Zc7Wuyp5PA__t9AO8W5Bc'
}


def download(dest, file_id):

    gdd.download_file_from_google_drive(file_id=file_id, dest_path=dest, unzip=True)
    os.remove(dest)


if __name__ == '__main__':

    # parse cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest', type=str, default='./datasets', help="""directory where to download the datasets""")
    parser.add_argument('--datasets', type=str, default=['pokemon', 'celeba', 'stl-10'], help="""which datasets to download""")

    args = parser.parse_args()
    datasets = [args.datasets] if isinstance(args.datasets, str) else args.datasets
    dest = args.dest

    # download datasets
    for dataset in datasets:
        filename = os.path.join(dest, dataset, f'{dataset}.zip')
        id = file_ids[dataset]
        download(filename, id)
    print('all done!')

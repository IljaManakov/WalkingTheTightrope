import torch as pt
from torch.utils.data import DataLoader
from datasets import Representations
import argparse


def get_codes(dataloader, model):

    codes = []
    labels = []
    total = len(dataloader)
    with pt.no_grad():
        for i, batch in enumerate(dataloader):
            labels.append(batch[1:])
            inputs = batch[0]
            c = model.encode(inputs)
            codes.append(c.cpu())
            print(f'\rprocessed batch {i+1} of {total}:\t{100*(i+1)/total:2}%', end='')
    print('')
    codes = pt.cat(codes)
    labels = [pt.cat(l) for l in list(zip(*labels))]
    samples = list(zip(codes, *labels))

    return samples


if __name__ == '__main__':

    # parse cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, required=True,
                        help='filename of the dataset .pt file to export representations from')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='filename of the model which will be used in the export')
    parser.add_argument('--output', '-o', type=str, default='representations.pt',
                        help='filename of the output file that will be created after the export')
    parser.add_argument('--cuda', default=False, action='store_true',
                        help='run export on gpu')
    parser.add_argument('--batch_size', '-b', type=int, default=128)
    args = parser.parse_args()

    # init model and data
    model = pt.load(args.model).cuda() if args.cuda else pt.load(args.model)
    transform = lambda x: [i.cuda() for i in x] if args.cuda else lambda x: x
    dataset = Representations(args.dataset, fraction=1, shuffle=0, transformation=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # save representations
    codes = get_codes(dataloader, model)
    pt.save(codes, args.output)


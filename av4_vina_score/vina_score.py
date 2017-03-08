import os, sys
import numpy as np
from eval import evaluation
from option import get_parser

FLAGS = None

def test():
    v = evaluation(FLAGS.receptor, FLAGS.ligand, FLAGS.debug, FLAGS.log)
    v.eval()

def landscape():
    v = evaluation(FLAGS.receptor, FLAGS.ligand, FLAGS.debug, FLAGS.log)
    shift_coords = [[x, y, z] for x in np.linspace(-1, 1, 3) for y in np.linspace(-1, 1, 21) for z in
                    np.linspace(-1, 1, 21)]
    landscape = []
    for shift in shift_coords:
        v.set_transform(shift=shift)
        e = v.eval()
        landscape.append(shift + [e])

    np.save('energy_map.npy', np.asarray(landscape))

def main():
    landscape()
    #test()

if __name__ == '__main__':

    parser = get_parser()
    FLAGS, unparsed = parser.parse_known_args()
    main()

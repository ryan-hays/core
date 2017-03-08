import os,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from collections import namedtuple
from option import visualize_parser

FLAGS = None
Axis = namedtuple('Axis',['label','dim'])

class Draw:

    def __init__(self,energy_map,dest_folder='images'):

        try:
            self.coords = np.load(energy_map)
        except Exception as e:
            print e
            exit(1)

        self.set_fix_axis()

        count = 0
        while os.path.exists(os.path.abspath(dest_folder+'_'+str(count))):
            count +=1
        os.mkdir(dest_folder+'_'+str(count))
        self.dest_folder = dest_folder+'_'+str(count)

    def set_fix_axis(self,axis='X'):
        if axis.upper() == 'X':
            self.fix = Axis('X',0)
            self.expand=[Axis('Y',1),
                         Axis('Z',2)]

        elif axis.upper() == 'Y':
            self.fix = Axis('Y', 1)
            self.expand = [Axis('X', 0),
                           Axis('Z', 2)]

        elif axis.upper() == 'Z':
            self.fix = Axis('Z', 2)
            self.expand = [Axis('X', 0),
                           Axis('Y', 1)]

    def save_image(self,sample_on_fix_axis=10):

        fixed_axis_value = np.unique(self.coords[:,self.fix.dim])
        sample_index = np.unique(np.linspace(0,len(fixed_axis_value)-1,sample_on_fix_axis).astype(int))
        sample_value = fixed_axis_value[sample_index]

        for value in sample_value:
            slice_on_fixed_axis = self.coords[self.coords[:,self.fix.dim]==value]
            edge_size = np.sqrt(slice_on_fixed_axis.shape[0]).astype(int)
            X = slice_on_fixed_axis[:, self.expand[0].dim].reshape(edge_size, edge_size)
            Y = slice_on_fixed_axis[:, self.expand[1].dim].reshape(edge_size, edge_size)
            Z = slice_on_fixed_axis[:, -1].reshape(edge_size,edge_size)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
            cset = ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.coolwarm)
            cset = ax.contour(X, Y, Z, zdir='x', offset=np.min(X), cmap=cm.coolwarm)
            cset = ax.contour(X, Y, Z, zdir='y', offset=np.max(Y), cmap=cm.coolwarm)

            ax.set_xlabel(self.expand[0].label)
            ax.set_ylabel(self.expand[1].label)
            ax.set_zlabel('Energy')

            image_name = '{}_{}.svg'.format(self.fix.label,"{:.3f}".format(value).replace('-','m').replace('.','_'))
            plt.savefig(os.path.join(self.dest_folder, image_name),transparent=True)
            plt.close(fig)

def main():
    draw = Draw(FLAGS.input_file,FLAGS.dest_folder)
    draw.set_fix_axis(FLAGS.fixed_axis)
    draw.save_image(FLAGS.images_num)

if __name__ == '__main__':

    parser = visualize_parser()
    FLAGS, unparsed = parser.parse_known_args()
    main()

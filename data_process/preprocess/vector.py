import prody as protein
import numpy as np

'''
Background: It will be much easier to use 4*4 matrix instead of 3*3 matrix to do transformation.
This will help when the program needs to judge whether the ligand is in rotated boxes, (in case of loss of information)

To be simple , all matrix is right-associative i.e.
[x',y',z',1] = [x,y,z,1]*T

T is transformation matrix
'''
def rotation_matrix_by_x(theta):
    '''
    just return the rotation matrix with clockwise theta degree along x-axis
    :param coord:
    :param theta:
    :return:
    '''
    return np.matrix(
        [[1,0,0,0],
         [0,np.cos(theta),-np.sin(theta),0],
         [0,np.sin(theta),np.cos(theta),0],
         [0,0,0,1]])

def rotation_matrix_by_y(theta):
    '''
    just return the rotation matrix with clockwise theta degree along y-axis
    :param coord:
    :param theta:
    :return:
    '''
    return np.matrix(
        [[np.cos(theta),0,np.sin(theta),0],
         [0,1,0,0],
         [-np.sin(theta),0,np.cos(theta),0],
         [0,0,0,1]])

def rotation_matrix_by_z(theta):
    '''
    just return the rotation matrix with clockwise theta degree along z-axis
    :param coord:
    :param theta:
    :return:
    '''
    return np.matrix(
        [[np.cos(theta),-np.sin(theta),0,0],
         [np.sin(theta),np.cos(theta),0,0],
         [0,0,1,0]
         [0,0,0,1]])


def transition_matrix(coord):
    '''

    :param coord:
    :param shift:
    :return:
    '''
    return np.matrix(
        [[1,0,0,0],
         [0,1,0,0],
         [0,0,1,0],
         [coord[0],coord[1],coord[2],1]
         ]
    )

def get_rotation_marix_along_anyaxis(center,rotation):
    '''
    There is a vector as rotation axis, which starts at center and
    according to following steps to be that direction:
    0. It originally point along +z-axis
    1. It rotate clockwisely with rotation[0] degree along x-axis
    2. It rotate clockwisely with rotation[1] degree along y-axis
    3. It rotate clockwisely with rotation[2] degree along z-axis

    (P.S. 6-freedom)

    :param center: center vector (used in transition transformation)
    :param rotation: rotation vector (used in rotation transformation)
    :return:
    '''
    T = transition_matrix(-center)
    T_inv= transition_matrix(center)
    R_x = rotation_matrix_by_x(rotation[0])
    R_x_inv = rotation_matrix_by_x(-rotation[0])
    R_y = rotation_matrix_by_y(rotation[1])
    R_y_inv = rotation_matrix_by_y(-rotation[1])
    R_z = rotation_matrix_by_z(rotation[2])
    return reduce(np.dot, [T,R_x,R_y,R_z,R_y_inv,R_x_inv,T_inv])

def get_transformation_inv_fromoldtonew(new_xcoord,new_ycoord,new_zcoord,transition):
    '''

    :param new_xcoord: (1,0,0) -> (ux,uy,uz)
    :param new_ycoord: (0,1,0) -> (vx,vy,vz)
    :param new_zcoord: (0,0,1) -> (wx,wy,wz)
    :param transition: **very important** this is used to align the start (may be down-left coordinate after rotation)
    because we need vectors instead of absolute coordinate to calculate
    i.e. (a*ux+transition[0],a*uy+transition[1],a*uz+transition[2]) might represent one edge of a cubic box.
    :return: **Inverse matrix** of this transformation
    '''
    T_inv = transition_matrix(-transition)
    U= np.matrix([new_xcoord,new_ycoord,new_zcoord])
    return np.dot(T_inv*U.transpose())

def get_coord_after_transformation(coord,transformation_matrix):
    return np.dot(coord+[1],transformation_matrix)[0:2]


def in_cubic_box(coords, Box, BOXsize= 20):
    '''
    This is the function that used to judge whether the coords are in the cubix box or not
    :param coords:
    :param center:
    :param BOXsize:
    :return:
    '''
    pass

class Box:
    center= [0,0,0]
    Boxsize= 20
    down_left = [0,0,0]

    # resolution of box
    Boxrange= 1
    Boxnum =  int(np.ceil(Boxsize+0.01 / Boxrange))

    rotation = [0,0,0]

    def __init__(self,**kwargs):
        if 'center' in kwargs:
            self.center= kwargs['center']
        if 'center' in kwargs:
            self.center = kwargs['center']
        if 'center' in kwargs:
            self.center = kwargs['center']
        if 'center' in kwargs:
            self.center = kwargs['center']


class vector_generator:
    receptor= None

    def __init__(self,receptor_filename):
        '''
        :param receptor_filename: source pdb file or .gz file that can be opened and parsed
        '''

        try:
            parse = protein.parsePDB(receptor_filename)
        except:
            print 'Cannot parse file %s, please check your input' % receptor_filename.split('/')[-1]
            return False

        self.receptor= parse.select('protein')

        if parse.select('nucleic') is not None:
            print 'This program does not support nucleic cocrystal structure for now.'
            return False




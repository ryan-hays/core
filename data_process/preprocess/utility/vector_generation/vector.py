import prody as protein
import numpy as np
import matplotlib as pl
import os

'''
Background: It will be much easier to use 4*4 matrix instead of 3*3 matrix to do transformation.
This will help when the program needs to judge whether the ligand is in rotated boxes, (in case of loss of information)

To be simple , all matrix is right-associative i.e.
[x',y',z',1] = [x,y,z,1]*T

(sometimes [x',y',z',1].transpose() = T*[x,y,z,1].transpose() is also a solution)

T is transformation matrix
'''

def rotation_matrix_by_x(theta):
    '''
    just return the rotation matrix with clockwise theta degree along x-axis
    :param coord:
    :param theta:
    :return:
    '''
    return np.array(
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
    return np.array(
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
    return np.array(
        [[np.cos(theta),-np.sin(theta),0,0],
         [np.sin(theta),np.cos(theta),0,0],
         [0,0,1,0],
         [0,0,0,1]])


def transition_matrix(coord):
    '''

    :param coord:
    :param shift:
    :return:
    '''
    return np.array(
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

def get_transformation_inv_fromoldtonew(new_xcoord,new_ycoord,new_zcoord,origin_coord):
    '''

    :param new_xcoord: (1,0,0) -> (ux,uy,uz)
    :param new_ycoord: (0,1,0) -> (vx,vy,vz)
    :param new_zcoord: (0,0,1) -> (wx,wy,wz)
    :param origin_coord: **very important** this is used to align the start (may be down-left coordinate after rotation)
    because we need vectors instead of absolute coordinate to calculate
    i.e. (a*ux+transition[0],a*uy+transition[1],a*uz+transition[2]) might represent one edge of a cubic box.
    :return: **Inverse matrix** of this transformation
    '''
    T_inv = transition_matrix(-origin_coord)
    U= np.array([[new_xcoord[0],new_xcoord[1],new_xcoord[2],0],
                  [new_ycoord[0],new_ycoord[1],new_ycoord[2],0],
                  [new_zcoord[0],new_zcoord[1],new_zcoord[2],0],
                  [0,0,0,1]],dtype=float)
    return np.dot(T_inv,U.transpose())

def get_coord_after_transformation(coord,transformation_matrix):
    '''
    Get absolute coordinate in original coordinate system (before rotation)
    :param coord:
    :param transformation_matrix:
    :return:
    '''
    extend_coord = np.resize(coord,4)
    extend_coord[3]=1
    return (np.dot(extend_coord,transformation_matrix)[0:3])

def get_zyx_position(coord,length):
    return coord[2]*length*length + coord[1]*length + coord[0]


class Box:
    center= np.array([10,10,10])
    Boxsize= 20
    down_left = np.array([0,0,0])
    x_axis= np.array([1,0,0])
    y_axis= np.array([0,1,0])
    z_axis= np.array([0,0,1])

    # resolution of box
    Boxrange= 1
    Boxnum =  int(np.ceil(Boxsize / Boxrange))

    rotation = [0,0,0]
    rotated= False
    transition_matrix= None

    def __init__(self,**kwargs):
        if 'center' in kwargs:
            self.center= np.array(kwargs['center'],dtype=float)
        if 'Boxsize' in kwargs:
            self.Boxsize = kwargs['Boxsize']
        if 'Boxrange' in kwargs:
            self.Boxrange = kwargs['Boxrange']
        self.down_left= self.center-np.array([self.Boxsize/2,self.Boxsize/2,self.Boxsize/2])
        self.Boxnum=  int(np.ceil(self.Boxsize/ self.Boxrange))

    def change_size(self,Boxsize,Boxrange=1):
        self.Boxsize= Boxsize
        self.Boxrange= Boxrange
        self.Boxnum = int(np.ceil(Boxsize/Boxrange))

    def change_manually(self,x_vec,y_vec,z_vec,center):
        self.x_axis = x_vec
        self.y_axis = y_vec
        self.z_axis = z_vec
        self.center = center
        self.down_left = self.center- self.Boxsize/2 * (x_vec+y_vec+z_vec)

    def set_default(self,center,Boxsize,Boxrange):
        self.center = np.array(center,dtype=float)
        self.Boxsize = Boxsize
        self.Boxrange = Boxrange

        self.down_left = self.center - np.array([self.Boxsize / 2, self.Boxsize / 2, self.Boxsize / 2])
        self.Boxnum = int(np.ceil(self.Boxsize / self.Boxrange))
        self.x_axis = np.array([1, 0, 0])
        self.y_axis = np.array([0, 1, 0])
        self.z_axis = np.array([0, 0, 1])

        self.rotation = [0, 0, 0]
        self.rotated = False
        self.transition_matrix = np.ones(4)
        self.coordinate_shift_matrix = np.ones(4)

    def transform(self,rotation,**kwargs):
        '''

        :param rotation: degrees to rotate , always [x,y,z] x for 'along x-axis clockwisely', so on so forth
        :param kwargs: center : new center of box
                        transition : shift from old box center
                        note only one can be put as input
        :return:
        '''
        if 'center' in kwargs:
            if 'transition' in kwargs:
                print 'This will cause ambiguity. Give either new center or shift from the old center'
                return
            else:
                self.center= kwargs['center']
        else:
            if 'transition' in kwargs:
                self.center += np.array(kwargs['transition'])
        T= get_rotation_marix_along_anyaxis(self.center,rotation)
        #print T
        self.rotated = True
        if self.transition_matrix is None:
            self.transition_matrix = T
        else:
            self.transition_matrix *= T

        new_down_left = get_coord_after_transformation(self.down_left,T)
        self.x_axis = get_coord_after_transformation(self.down_left+self.x_axis, T)-new_down_left
        self.y_axis = get_coord_after_transformation(self.down_left+self.y_axis, T)-new_down_left
        self.z_axis = get_coord_after_transformation(self.down_left+self.z_axis, T)-new_down_left

        self.down_left = new_down_left
        #print self.x_axis
        #print self.y_axis
        #print self.z_axis
        #print self.down_left

        self.coordinate_shift_matrix = \
            get_transformation_inv_fromoldtonew(self.x_axis,
                                                self.y_axis,
                                                self.z_axis,
                                                self.down_left)

    def find_lattice_coord(self,coord):
        '''

        :param coord: [x,y,z] all be interger to represent grid (lattice) in box (down_lefy is [0,0,0])
        :return: real coords in xyz-coordinatesystem (Absolute one)
        '''
        if coord[0]<0 or coord[0]>=self.Boxsize:
            print 'x-coordinate out of border!'
            return
        if coord[1]<0 or coord[1]>=self.Boxsize:
            print 'y-coordinate out of border!'
            return
        if coord[2]<0 or coord[2]>=self.Boxsize:
            print 'z-coordinate out of border!'
            return

        #print self.down_left + np.dot(coord,np.array([self.x_axis,self.y_axis,self.z_axis]))
        return self.down_left + np.dot(coord,np.array([self.x_axis,self.y_axis,self.z_axis]))

    def get_lattice_coord(self,coords):
        '''
        coords is one coordinate in old coordinate system, this function will return new coordinate
        and return the shift from the down_left point
        :param coords:
        :return:
        '''
        new_coord=get_coord_after_transformation(coords, self.coordinate_shift_matrix)
        #print new_coord
        return np.round(new_coord)

    def in_cubic_box(self,coords):
        '''
        This is the function that used to judge whether the coords are in the cubix box or not
        :param coords:
        :param center:
        :param BOXsize:
        :return:
        '''
        relative_coords= self.get_lattice_coord(coords)
        return (0<=relative_coords[0]<self.Boxsize and 0<=relative_coords[1]<self.Boxsize and 0<=relative_coords[2]<self.Boxsize)

    def _box_visualize(self):
        '''
        For fun or check
        To be continued
        :return: the plot of box (in Absolute coordinatesystem)
        '''

    def self_test(self):
        rotation = np.random.random_sample(3) * np.pi / 4
        transition = (np.random.random_sample(3) - [0.25] * 3) * 2 * [0.5,0.5,0.5]
        print 'Rotation:'
        print rotation
        print 'Transition:'
        print transition
        print 'Begin Testing:'

        origin_down_left = self.down_left
        origin_x = self.x_axis
        origin_y = self.y_axis
        origin_z = self.z_axis


        self.transform(rotation, transition=transition)

        if self.rotated==True:
            print 'Already rotated. Now let us pick some point on lattice randomly'

        NUM= 10
        sample=[]
        for i in range(NUM):
            id = str(i+1)
            lattice= np.random.randint(0,self.Boxsize-1,3)
            sample.append(lattice)
            print 'Pick lattice : ' + id
            print 'lattice coordinate: '+ str(lattice)

            coord = origin_down_left + np.dot(lattice,np.array([origin_x,origin_y,origin_z]))

            print 'original point is : '+ str(coord)
            new_coord =self.find_lattice_coord(lattice)
            print 'new coordinate of lattice {}'.format(new_coord)
            print 'the inner function shows the new coordinate is for lattice {}'.format(self.get_lattice_coord(new_coord))


        print '\nTest ended.'

# Tag for hetero atoms
HETERO_PART = 'L'
# Tag for protein atoms
PROTEIN_PART = 'P'

class vector_generator:
    receptor= None
    heterodict={}
    boxdict={}
    Boxsize=20
    Boxrange=1

    def __init__(self,receptor_filename,Boxsize=20,Boxrange=1):
        '''
        :param receptor_filename: source pdb file or .gz file that can be opened and parsed
        '''

        try:
            parse = protein.parsePDB(receptor_filename)
        except:
            print 'Cannot parse file %s, please check your input' % receptor_filename.split('/')[-1]
            return

        self.receptor= parse.select('protein')

        if parse.select('nucleic') is not None:
            print 'This program does not support nucleic cocrystal structure for now.'
            return

        hetero = parse.select('(hetero and not water) or resname ATP or resname ADP')

        if hetero is None:
            return

        for pick_one in protein.HierView(hetero).iterResidues():
            # less than 3 atoms may be not ok
            if pick_one.numAtoms() <= 3:
                continue

            ResId = pick_one.getResindex()
            self.heterodict[ResId]=pick_one

            # Set up a new box class here
            self.boxdict[ResId]= Box(center=protein.calcCenter(pick_one).getCoords(),Boxsize=Boxsize,Boxrange=Boxrange)

        self.Boxsize=Boxsize
        self.Boxrange=Boxrange

    def set_ligand_from_file(self,ligand_filename):
        '''
        :param ligand_filename: source pdb file or mol2 file that can be opened and parsed
         pdb might be better because mol2 will loss atom_name (most only indicate O/N/C)
        :return:
        '''
        try:
            suffix = ligand_filename.split('.')[-1]
        except:
            print 'ligand file parse error! check the address '+ ligand_filename
            return 'NA'

        try:
            if suffix!='pdb':
                #Note it will only try to parse first one
                os.system('babel -imol2 %s -opdb temp.pdb'% ligand_filename)
                ligand = protein.parsePDB('temp.pdb')
            else:
                ligand = protein.parsePDB(ligand_filename)
        except:
            print 'Only support pdb or mol2 format, please check your file. It will cast a parse Error!'
            return 'NA'

        ResId= ''.join(map(lambda xx:(hex(ord(xx))[2:]),os.urandom(16)))
        self.heterodict[ResId] = ligand
        self.boxdict[ResId] = Box(center=protein.calcCenter(ligand), Boxsize=self.Boxsize, Boxrange=self.Boxrange)
        return ResId

    def generate_vector_from_file(self,ligand_filename,try_threshold=200,shift_threshold=[1,1,1],OUT=False,verbose=False):
        '''
        the main program to generate vectors from a ligand in a specific receptor:
        pipeline:
        1. do transformation to box
        2. see if the ligand is still in the box, if not do 1 again until try over try_threshold times, otherwise move to 3
        3. select inbox receptor part
        4. bundle the vector and return

        By default, the center is average of each atom's position. But the program can do little shift to this point because
        when the box is doing rotation, sometimes some part of ligand might out of the range, it needs adjustment.

        :param ligand_filename: where source file can be read and parsed , pdb or mol2 format only
        :param try_threshold: if random transition failed over this many times, the program will inform an error
        instead of return a valid vector. Note the first half trial times will only concern rotation. Then shift.
        :param shift_threshold: the absolute value of transition transform will not exceed this threshold
        :return: either a vector in random box 'on the fly' or get an Error (False, i.e.)
        '''
        ResId= self.set_ligand_from_file(ligand_filename)
        if ResId=='NA':
            return False, [0]*8000

        ligand = self.heterodict[ResId]

        Box = self.boxdict[ResId]
        ligand_center = protein.calcCenter(ligand)

        # only do rotation
        for iteration_step in range(try_threshold/2):
            # Now try to do one rotation

            rotation=np.random.random_sample(3)*np.pi/4
            Box.transform(rotation)
            Tag= True
            for atom in ligand.iterAtoms():
                coord = atom.getCoords()
                if Box.in_cubic_box(coord)==False:
                    Tag= False
            if Tag == False:
                if verbose==True:
                    print 'Try %s : rotation failed to contain ligand.' % str(iteration_step)
                continue


            #Now select potential receptor part
            residues = self.receptor.select('protein and same residue as within {} of center'
                                            .format(int(np.ceil(np.sqrt(3)*self.Boxsize/2))), center=ligand_center)

            if residues is None:
                if verbose==True:
                    print 'Try %s : This box has no protein atoms nearby' % str(iteration_step)
                continue
            num_vector = [''] * (self.Boxsize**3)

            # First add ligand part
            for atom in ligand.iterAtoms():
                coord = atom.getCoords()
                if Box.in_cubic_box(coord)==False:
                    continue
                lattice = Box.get_lattice_coord(coord)
                pos = get_zyx_position(lattice,self.Boxsize)
                pos = int(np.round(pos))
                tag = atom.getName() + '_' + str(HETERO_PART)
                if verbose==True:
                    print 'We found an atom at {} with tag {}'.format(pos,tag)
                num_vector[pos]= tag

            # Then add receptor part
            for atom in residues.iterAtoms():
                coord = atom.getCoords()
                if Box.in_cubic_box(coord)==False:
                    continue
                lattice = Box.get_lattice_coord(coord)
                pos = get_zyx_position(lattice,self.Boxsize)
                pos = int(np.round(pos))
                tag = atom.getName() + '_' + str(PROTEIN_PART)

                if verbose==True:
                    print 'We found an atom at {} with tag {}'.format(pos, tag)
                if num_vector[pos]!='':
                    if verbose==True:
                        print 'Now we have a conflict at '+ str(lattice) +', atom %s will be put with ligand' \
                                                                      '\'s atom together'%(atom.getName())
                    num_vector[pos]+='|' + tag
                else:
                    num_vector[pos]= tag

            if OUT==True:
                #For verbose purpose
                ligand.select('all').setResnums(9999)
                protein.saveAtoms(residues,filename='temp.ag.npz')
                atomgroup = protein.loadAtoms('temp.ag.npz')
                protein.writePDB('debug.pdb',ligand+atomgroup)
            return True, num_vector

        # try iteration and shift altogether
        for iteration_step in range(try_threshold/2,try_threshold):
            # Now try to do one rotation

            rotation = np.random.random_sample(3) * np.pi / 4
            transition = (np.random.random_sample(3)-[shift_threshold/2]*3) * 2 * shift_threshold
            Box.transform(rotation,transition=transition)
            Tag = True
            for atom in ligand.iterAtoms():
                coord = atom.getCoords()
                if Box.in_cubic_box(coord) == False:
                    Tag = False
            if Tag == False:
                if verbose == True:
                    print 'Try %s : rotation plus transition failed to contain ligand.' % str(iteration_step)
                continue

            # Now select potential receptor part
            residues = self.receptor.select('protein and same residue as within {} of center'
                                            .format(int(np.ceil(np.sqrt(3) * self.Boxsize / 2))), center=Box.center)

            if residues is None:
                if verbose == True:
                    print 'Try %s : This box has no protein atoms nearby' % str(iteration_step)
                continue
            num_vector = [''] * (self.Boxsize ** 3)

            # First add ligand part
            for atom in ligand.iterAtoms():
                coord = atom.getCoords()
                if Box.in_cubic_box(coord) == False:
                    continue
                lattice = Box.get_lattice_coord(coord)
                pos = get_zyx_position(lattice, self.Boxsize)
                pos = int(np.round(pos))
                tag = atom.getName() + '_' + str(HETERO_PART)
                print 'We found an atom at {} with tag {}'.format(pos, tag)
                num_vector[pos] = tag

            # Then add receptor part
            for atom in residues.iterAtoms():
                coord = atom.getCoords()
                if Box.in_cubic_box(coord) == False:
                    continue
                lattice = Box.get_lattice_coord(coord)
                pos = get_zyx_position(lattice, self.Boxsize)
                pos = int(np.round(pos))
                tag = atom.getName() + '_' + str(PROTEIN_PART)
                if verbose == True:
                    print 'We found an atom at {} with tag {}'.format(pos, tag)
                if num_vector[pos] != '':
                    if verbose ==True:
                        print 'Now we have a conflict at ' + str(lattice) + ', atom %s will be put with ligand' \
                                                                        '\'s atom together' % (atom.getName())
                    num_vector[pos] += '|' + tag
                else:
                    num_vector[pos] = tag

            if OUT==True:
                #For verbose purpose
                ligand.select('all').setResnums(9999)
                protein.writePDB('debug.pdb',ligand+residues)
            return True, num_vector

        if verbose == True:
            print 'try enough times and all failed, consider a smaller ligand instead.'
        return False,[0]*8000

    def __del__(self):
        if os.path.exists('temp.ag.npz'):
            os.remove('temp.ag.npz')
        if os.path.exists('temp.pdb'):
            os.remove('temp.pdb')

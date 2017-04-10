import tensorflow as tf
import numpy as np
from deepVS_input import *

#---------------------------------HYPERPARAMETERS---------------------------------#

#atom type embedding size
d_atm = 200
#amino acid embedding size
d_amino = 200
#charge embedding size
d_chrg = 200
#distance embedding size
d_dist = 200
#number convolutional filters
cf = 400
#number hidden units
h = 50
#learning rate
l = 0.075
#number of neighbor atoms from ligand
k_c = 6
#number of neighbor atoms from protein
#k_p = 0
#number of atom types
ATOM_TYPES = 7
#number of distance bins
DIST_BINS = 18
DIST_INTERVAL = 0.3

#-------------------------------LAYER CONSTRUCTION--------------------------------#

# telling tensorflow how we want to randomly initialize weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.005)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var, name):
    """attaches a lot of summaries to a tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)

def embed_layer(layer_name, input, num_atoms):
	"""transforms the mapped z into feature vectors and returns the resulting tensor"""
	#input is standard matrix of shape m * k_c * 2
	#output is tensor of shape k_c * d_atm+d_dist * m
	with tf.name_scope(layer_name):
		with tf.name_scope('atom_weights'):
			W_atom = weight_variable([ATOM_TYPES, d_atm])
		with tf.name_scope('dist_weights'):
			W_dist = weight_variable([DIST_BINS, d_dist])
	for atom_index in range(len(input)):
		for neighbor_index in range(len(input[0])):
			atom_type = input[atom_index][0][0]
			dist_bin = input[atom_index][0][1]
			if neighbor_index == 0:
				face = tf.concat([tf.gather_nd(W_atom, [[atom_type]]), tf.gather_nd(W_dist, [[dist_bin]])], axis=1)
			else:
				#get a 1x400 feature embedding for a neighbor
				one_neighbor_features = tf.concat([tf.gather_nd(W_atom, [[atom_type]]), tf.gather_nd(W_dist, [[dist_bin]])], axis=1)
				#turns face into k_c * 400
				face = tf.concat([face, one_neighbor_features], 0)
		if atom_index == 0:
			embedded_input = tf.reshape(face, [k_c, d_atm+d_dist, 1])
		else:
			#turns embedded input into k_c * 400 * m tensor
			embedded_input = tf.concat([embedded_input, tf.reshape(face, [k_c, d_atm+d_dist, 1])], 2)
	#gives embedded_input the depth dimension and batch size
	return tf.reshape(embedded_input, [1, k_c, d_atm+d_dist, num_atoms, 1])

def conv_layer(layer_name, input_tensor, filter_size, strides=[1,1,1,1,1], padding='SAME'):
	"""makes a simple face convolutional layer"""
	with tf.name_scope(layer_name):
		with tf.name_scope('weights'):
			W_conv = weight_variable(filter_size)
			variable_summaries(W_conv, layer_name + '/weights')
		with tf.name_scope('biases'):
			b_conv = bias_variable([filter_size[3]])
			variable_summaries(b_conv, layer_name + '/biases')
		h_conv = tf.nn.conv3d(input_tensor, W_conv, strides=strides, padding=padding) + b_conv
		tf.summary.histogram(layer_name + '/pooling_output', h_conv)
	print layer_name,"output dimensions:", h_conv.get_shape()
	return h_conv

def pool_layer(layer_name, input_tensor, num_atoms):
	"""makes a max pool layer that returns the max of each column"""
	with tf.name_scope(layer_name):
		h_pool = tf.nn.max_pool3d(input_tensor, ksize=[1,1,1,num_atoms,1], strides=[1,1,1,1,1], padding='VALID')
		tf.summary.histogram(layer_name + '/max_pooling_output', h_pool)
	print layer_name, 'output dimensions:', h_pool.get_shape()
	return h_pool

def fc_layer(layer_name,input_tensor,output_dim):
	"""makes a simple fully connected layer"""
	input_dim = int((input_tensor.get_shape())[1])

	with tf.name_scope(layer_name):
		weights = weight_variable([input_dim, output_dim])
		variable_summaries(weights, layer_name + '/weights')
	with tf.name_scope('biases'):
		biases = bias_variable([output_dim])
		variable_summaries(biases, layer_name + '/biases')
	with tf.name_scope('Wx_plus_b'):
		h_fc = tf.matmul(input_tensor, weights) + biases
		tf.summary.histogram(layer_name + '/fc_output', h_fc)
	print layer_name, "output dimensions:", h_fc.get_shape()
	return h_fc

#-----------------------------------NETWORK----------------------------------------#

#TODO get ligand_atoms and ligand_coords in the right format
#Assuming the right input formats to class Z, calling
#Z().z returns a matrix of usable input data


class Z(object):

	def atom_dictionary(self):
		ATM = {}

		ATM["h"] = 1; ATM["h1"] = 1; ATM["h2"] = 1; ATM["h3"] = 1; ATM["h4"] = 1; ATM["h5"] = 1; ATM["h6"] = 1; ATM["h7"] = 1; ATM["h8"] = 1;
		ATM["h1*"] = 1; ATM["h2*"] = 1; ATM["h3*"] = 1; ATM["h4*"] = 1; ATM["h5*"] = 1; ATM["h6*"] = 1; ATM["h7*"] = 1; ATM["h8*"] = 1;
		ATM["hg"] = 1; ATM["hxt"] = 1; ATM["hz1"] = 1; ATM["hz2"] = 1; ATM["he2"] = 1; ATM["d"] = 1

		ATM["c"] = 2; ATM["c1"] = 2; ATM["c2"] = 2; ATM["c3"] = 2; ATM["c4"] = 2; ATM["c5"] = 2; ATM["c6"] = 2; ATM["c7"] = 2; ATM["c8"] = 2;
		ATM["c1*"] = 2; ATM["c2*"] = 2; ATM["c3*"] = 2; ATM["c4*"] = 2; ATM["c5*"] = 2; ATM["c7*"] = 2; ATM["c8*"] = 2;
		ATM["cb"] = 2; ATM["ca"] = 2; ATM["ce"] = 2; ATM["cg"] = 2; ATM["cd"] = 2; ATM["cd1"] = 2; ATM["cd2"] = 2;

		ATM["n"] = 3; ATM["n1"] = 3; ATM["n2"] = 3; ATM["n3"] = 3; ATM["n4"] = 3; ATM["n5"] = 3; ATM["n6"] = 3; ATM["n7"] = 3; ATM["n8"] = 3; ATM["nz"] = 3;

		ATM["o"] = 4; ATM["o1"] = 4; ATM["o2"] = 4; ATM["o3"] = 4; ATM["o4"] = 4; ATM["o5"] = 4; ATM["o6"] = 4; ATM["o7"] = 4; ATM["o8"] = 4;
		ATM["o1*"] = 4; ATM["o2*"] = 4; ATM["o3*"] = 4; ATM["o4*"] = 4; ATM["o5*"] = 4; ATM["o6*"] = 4; ATM["o7*"] = 4; ATM["o8*"] = 4;
		ATM["oe1"] = 4; ATM["oe2"] = 4; ATM["cd1"] = 4; ATM["oxt"] = 4;

		ATM["f"] = 5; ATM["cl"] = 5; ATM["i"] = 5; ATM["br"] = 5;

		ATM["p"] = 6; ATM["s"] = 6; # FIXME S is not really equal to P

		ATM["b"] = 7; ATM["xx"] = 7; ATM["mg"] = 7; ATM["zn"] = 7; ATM["fe"] = 7; ATM["se"] = 7; ATM["v"] = 7; ATM["sg"] = 7;
		ATM['ni'] = 7; ATM['co'] = 7; ATM['as'] = 7; ATM['ru']=7; ATM['mn'] = 7; ATM['mo'] = 7; ATM['re'] = 7; ATM['si'] = 7;
		return ATM

	def __init__(self, ligand_atoms, ligand_coords):
            """
            :param ligand_atoms: a set of all atoms in our ligand. in the form of characters
            :param ligand_coords: a dictionary mapping atoms to its coordinate as a tuple. for example "c": (0,0,0)
            :param kc: an int - the number of neighbors we want to consider

            self.atom_map is a dictionary mapping atoms to an integer
                for example "c": 2. Carbon gets mapped to 2
            self.z is a 3D matrix with dimensions [kc x 2 x m] where m is the total number of atoms in the complex
                each "face" of the matrix is the kc closest neighbors for an atom A. We have m of these faces - one for each atom
                Each row in a face is [neighbor, distance] - the neighbor to A, and its distance to A
                    we have kc rows in each face.
            """
		self.ligand_atoms = ligand_atoms
		self.ligand_coords = ligand_coords
		self.atom_map = self.atom_dictionary()
		self.z = self.build_z()

	def convert_coords_to_distances(self, start_atom):
		ligand_distances = {}
		atom_coord = self.ligand_coords[start_atom]
		for neighbor in self.ligand_coords:
			ligand_distances[neighbor] = self.distance(self.ligand_coords[neighbor], atom_coord)
		return ligand_distances

	def distance(self, coord1, coord2):
		x1, y1, z1 = coord1
		x2, y2, z2 = coord2
		return ((x1-x2) ** 2 + (y1-y2) ** 2 + (z1-z2) ** 2) ** 0.5

	def get_closest_atoms_and_distances(self, atom):
		distances = self.convert_coords_to_distances(atom)
		closest = {}
		for _ in range(k_c):
			closest_atom = min(distances, key=distances.get)
			closest[closest_atom] = distances[closest_atom]
			del distances[closest_atom]
		return closest

	def get_raw_z(self):
		#returns matrix of dimensions [k_c * 2 * m]
		raw_z = []
		for atom in self.ligand_atoms:
			kc_neighbors_dict = self.get_closest_atoms_and_distances(atom)
			kc_neighbors_list = [[neighbor, distance] for neighbor, distance in kc_neighbors_dict.items()]
			kc_neighbors_list.sort(key=lambda x: x[1]) #sort by distance
			raw_z.append(kc_neighbors_list)
		return raw_z


	def build_z(self):
		#returns matrix of dimensions [k_c * 2 * m]
		raw_z = self.get_raw_z()
		for atom_index in range(len(raw_z)): #iterate over atoms dimension (m)
			for neighbor_index in range(len(raw_z[0])): #iterate over neighbors (kc)
				atom, distance = raw_z[atom_index][neighbor_index]
				raw_z[atom_index][neighbor_index][0] = self.atom_map[atom]
				raw_z[atom_index][neighbor_index][1] = int(distance//DIST_INTERVAL + 1)
		return raw_z


def deepVS_net(ligand_atoms, ligand_coords, keep_prob):
	#gets a standard array of dimensions m * k_c * 2
	input = Z(ligand_atoms, ligand_coords).z
	#do the feature embedding to get a k_c * (d_atm + d_dist) * m TENSOR
	z = embed_layer('embed_layer', input, len(input))
	#convolutional layer - padding = 'VALID' prevents 0 padding
	z_conv = conv_layer('face_conv', input_tensor=z, filter_size=[k_c, d_atm+d_dist, 1, 1, cf], padding='VALID')
	#max pool along the columns (corresponding to each convolutional filter)
	z_pool = pool_layer(layer_name='pool_column', input_tensor=z_conv, num_atoms=len(input))
	#pool gives us batch*1*1*1*cf tensor; flatten it to get a tensor of length cf
	z_flattened = tf.reshape(z_pool, [-1, cf])
	#fully connected layer
	z_fc1 = fc_layer(layer_name='fc1', input_tensor=z_flattened, output_dim=h)
	#dropout
	#output layer
	z_output = fc_layer(layer_name='out_neuron', input_tensor=z_fc1, output_dim=2)
	return z_output



# sess = tf.Session()
# print(sess.run(ligand_atoms))
# time.sleep(100)

tligand_atoms = {'h', 'c', 'n', 'o', 'f', 'p', 'b', 'h2'}
tligand_coords = {'h': (0,0,0), 'c': (0,0,1), 'n': (0,2,0), 'o': (1,1,1), 'f': (2,2,3), 'p': (1.5, 1.5, 1.5), 'b': (2,0,0), 'h2': (1,1,1.5)}
test = deepVS_net(tligand_atoms, tligand_coords, 1)
print(test)

from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from av4_dcgan_ops import *
from utils import *

def conv_out_size_same(size, stride):
	return math.ceil(float(size) / float(stride))

class DCGAN(object):
	def __init__(self, sess, input_height=40, input_width=40, input_depth =40, is_crop=True,
		batch_size=64, sample_num = 64, output_height=64, output_width=64, output_depth = 64,
		y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
		gfc_dim=1024, dfc_dim=1024, c_dim=3, checkpoint_dir=None, sample_dir=None):
	# """

	# Args:
	# 	sess: TensorFlow session
	# 	batch_size: The size of batch. Should be specified before training.
	# 	y_dim: (optional) Dimension of dim for y. [None]
	# 	z_dim: (optional) Dimension of dim for Z. [100]
	# 	gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
	# 	df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
	# 	gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
	# 	dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
	# 	c_dim: (optional) Number of channels, which represent different atoms. [14]
	# """
		self.sess = sess
		self.is_crop = is_crop
		self.channels = (c_dim == 14)

		self.batch_size = batch_size
		self.sample_num = sample_num

		self.input_height = input_height
		self.input_width = input_width
		self.input_depth = input_depth

		self.output_height = output_height
		self.output_width = output_width
		self.output_depth = output_depth

		self.y_dim = y_dim
		self.z_dim = z_dim

		self.gf_dim = gf_dim
		self.df_dim = df_dim

		self.gfc_dim = gfc_dim
		self.dfc_dim = dfc_dim

		self.c_dim = c_dim

	    # batch normalization : deals with poor initialization helps gradient flow
		self.d_bn1 = batch_norm(name='d_bn1')
		self.d_bn2 = batch_norm(name='d_bn2')

		if not self.y_dim:
			self.d_bn3 = batch_norm(name='d_bn3')

			self.g_bn0 = batch_norm(name='g_bn0')
			self.g_bn1 = batch_norm(name='g_bn1')
			self.g_bn2 = batch_norm(name='g_bn2')

		if not self.y_dim:
			self.g_bn3 = batch_norm(name='g_bn3')



		self.checkpoint_dir = checkpoint_dir
		self.build_model()

	def build_model(self):
		if self.y_dim:
			self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

		image_dims = [self.input_height, self.input_height, self.c_dim]

		self.inputs = tf.placeholder(
			tf.int32, [self.batch_size] + image_dims, name='real_images')
		self.sample_inputs = tf.placeholder(
			tf.int32, [self.sample_num] + image_dims, name='sample_inputs')

		inputs = self.inputs
		sample_inputs = self.sample_inputs
		self.z = tf.placeholder(
			tf.float32, [None, self.z_dim], name='z')
		self.z_sum = histogram_summary("z", self.z)
	

		if self.y_dim:
			self.G = self.generator(self.z, self.y)
			self.D, self.D_logits = \
			self.discriminator(inputs, self.y, reuse=False)

			self.sampler = self.sampler(self.z, self.y)
			self.D_, self.D_logits_ = \
			self.discriminator(self.G, self.y, reuse=True)
		else:
			self.G = self.generator(self.z)
			self.D, self.D_logits = self.discriminator(inputs)

			self.sampler = self.sampler(self.z)
			self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

		self.d_sum = histogram_summary("d", self.D)
		self.d__sum = histogram_summary("d_", self.D_)
		self.G_sum = image_summary("G", self.G)

		self.d_loss_real = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(
				logits=self.D_logits, targets=tf.ones_like(self.D)))
		self.d_loss_fake = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(
				logits=self.D_logits_, targets=tf.zeros_like(self.D_)))
		self.g_loss = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(
				logits=self.D_logits_, targets=tf.ones_like(self.D_)))

		self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
		self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

		self.d_loss = self.d_loss_real + self.d_loss_fake

		self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
		self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

		t_vars = tf.trainable_variables()

		self.d_vars = [var for var in t_vars if 'd_' in var.name]
		self.g_vars = [var for var in t_vars if 'g_' in var.name]

		self.saver = tf.train.Saver()

	def discriminator(self, image, y=None, reuse=False):
		with tf.variable_scope("discriminator") as scope:
			if reuse:
				scope.reuse_variables()

				if not self.y_dim:
					h0 = lrelu(conv3d(image, self.df_dim, name='d_h0_conv'))
					h1 = lrelu(self.d_bn1(conv3d(h0, self.df_dim*2, name='d_h1_conv')))
					h2 = lrelu(self.d_bn2(conv3d(h1, self.df_dim*4, name='d_h2_conv')))
					h3 = lrelu(self.d_bn3(conv3d(h2, self.df_dim*8, name='d_h3_conv')))
					h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

					return tf.nn.sigmoid(h4), h4
				else:
					yb = tf.reshape(y, [self.batch_size, 1, 1, 1, self.y_dim])
					x = conv_cond_concat(image, yb)

					h0 = lrelu(conv3d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
					h0 = conv_cond_concat(h0, yb)

					h1 = lrelu(self.d_bn1(conv3d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
					h1 = tf.reshape(h1, [self.batch_size, -1])      
					h1 = concat([h1, y], 1)

					h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
					h2 = concat([h2, y], 1)

					h3 = linear(h2, 1, 'd_h3_lin')

					return tf.nn.sigmoid(h3), h3

	def generator(self, z, y=None):
		with tf.variable_scope("generator") as scope:
			if not self.y_dim:
				s_h, s_w, s_d = self.output_height, self.output_width, self.output_depth
				s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2) , conv_out_size_same(s_d, 2)
				s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
				s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
				s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

				# project `z` and reshape
				self.z_, self.h0_w, self.h0_b, self.h0_d = linear(
				z, self.gf_dim*8*s_h16*s_w16*s_d16, 'g_h0_lin', with_w=True)

				self.h0 = tf.reshape(
				self.z_, [-1, s_h16, s_w16, s_d16, self.gf_dim * 8])
				h0 = tf.nn.relu(self.g_bn0(self.h0))

				self.h1, self.h1_w, self.h1_b, self.h1_d = deconv3d(
				h0, [self.batch_size, s_h8, s_w8, s_d8, self.gf_dim*4], name='g_h1', with_w=True)
				h1 = tf.nn.relu(self.g_bn1(self.h1))

				h2, self.h2_w, self.h2_b, self.h2_d = deconv3d(
				h1, [self.batch_size, s_h4, s_w4, s_d4, self.gf_dim*2], name='g_h2', with_w=True)
				h2 = tf.nn.relu(self.g_bn2(h2))

				h3, self.h3_w, self.h3_b, self.h3_d = deconv3d(
				h2, [self.batch_size, s_h2, s_w2, s_d2, self.gf_dim*1], name='g_h3', with_w=True)
				h3 = tf.nn.relu(self.g_bn3(h3))

				h4, self.h4_w, self.h4_b, self.h4_d = deconv3d(
				h3, [self.batch_size, s_h, s_w, s_d, self.c_dim], name='g_h4', with_w=True)

				return tf.nn.tanh(h4)
			else:
				s_h, s_w, s_d = self.output_height, self.output_width, self.output_depth
				s_h2, s_h4 = int(s_h/2), int(s_h/4)
				s_w2, s_w4 = int(s_w/2), int(s_w/4)
				s_d2, s_d4 = int(s_d/2), int(s_d/4)

				# yb = tf.expand_dims(tf.expand_dims(y, 1),2)
				yb = tf.reshape(y, [self.batch_size, 1, 1, 1, self.y_dim])
				z = concat([z, y], 1)

				h0 = tf.nn.relu(
					self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
				h0 = concat([h0, y], 1)

				h1 = tf.nn.relu(self.g_bn1(
					linear(h0, self.gf_dim*2*s_h4*s_w4*s_d4, 'g_h1_lin')))
				h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, s_d4, self.gf_dim * 2])

				h1 = conv_cond_concat(h1, yb)

				h2 = tf.nn.relu(self.g_bn2(deconv3d(h1,
					[self.batch_size, s_h2, s_w2, s_d2, self.gf_dim * 2], name='g_h2')))
				h2 = conv_cond_concat(h2, yb)

				return tf.nn.sigmoid(
					deconv3d(h2, [self.batch_size, s_h, s_w, s_d, self.c_dim], name='g_h3'))
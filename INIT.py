from ops import *
from utils import *
from glob import glob
import time
import random
from tqdm import tqdm
from tensorflow.contrib.data import batch_and_drop_remainder
from sklearn.model_selection import train_test_split
import pickle

class INIT(object) :
    def __init__(self, sess, args):
        self.model_name = 'INIT'
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag

        self.epoch = args.epoch
        self.iteration = args.iteration

        self.gan_type = args.gan_type

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.num_style = args.num_style # for test
        self.guide_img = args.guide_img
        self.direction = args.direction

        self.img_h = args.img_h
        self.img_w = args.img_w
        self.img_ch = args.img_ch
        self.inst_w_w = args.inst_w

        self.init_lr = args.lr
        self.ch = args.ch

        """ Weight """
        # global
        self.gan_w = args.gan_w
        self.recon_x_w = args.recon_x_w
        self.recon_s_w = args.recon_s_w
        self.recon_c_w = args.recon_c_w
        self.recon_x_cyc_w = args.recon_x_cyc_w
        # instance
        self.gan_o_w = args.gan_o_w
        self.recon_o_w = args.recon_o_w
        self.recon_o_s_w = args.recon_o_s_w
        self.recon_o_c_w =args.recon_o_c_w
        self.recon_o_cyc_w = args.recon_o_cyc_w

        """ Generator """
        self.n_res = args.n_res
        self.mlp_dim = pow(2, args.n_sample) * args.ch # default : 256

        self.n_downsample = args.n_sample
        self.n_upsample = args.n_sample
        self.style_dim = args.style_dim

        """ """
        self.encode_parameters_share = True

        """ Discriminator """
        self.n_dis = args.n_dis
        self.n_scale = args.n_scale

        # self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        self.sample_dir = os.path.join(args.sample_dir, 'INIT_lsgan')
        check_folder(self.sample_dir)

        self.data_set = args.dataset
        self.data_folder = args.data_folder

        self.dataset_before_split = os.path.join(self.data_folder, 'data', 'all_data.pkl')
        self.dataset_path_trainA = os.path.join(self.data_folder, 'data', 'trainA.npy')
        self.dataset_path_trainB = os.path.join(self.data_folder, 'data', 'trainB.npy')
        self.dataset_path_testA = os.path.join(self.data_folder, 'data', 'testA.npy')
        self.dataset_path_testB = os.path.join(self.data_folder, 'data', 'testB.npy')


        # self.dataset_num = max(len(self.trainA_dataset), len(self.trainB_dataset))

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        # print("# max dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# iteration per epoch : ", self.iteration)
        print("# style in test phase : ", self.num_style)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)
        print("# Style dimension : ", self.style_dim)
        print("# MLP dimension : ", self.mlp_dim)
        print("# Down sample : ", self.n_downsample)
        print("# Up sample : ", self.n_upsample)

        print()

        print("##### Discriminator #####")
        print("# Discriminator layer : ", self.n_dis)
        print("# Multi-scale Dis : ", self.n_scale)

        print()

        print("##### Folder #####")
        print('sample directory : ', self.sample_dir)

    ##################################################################################
    # Encoder and Decoders
    ##################################################################################

    def Style_Encoder(self, x, reuse=False, scope='style_encoder'):
        # IN removes the original feature mean and variance that represent important style information
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse) :
            x = conv(x, channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv_0')
            x = relu(x)

            for i in range(2) :
                x = conv(x, channel*2, kernel=4, stride=2, pad=1, pad_type='reflect', scope='conv_'+str(i+1))
                x = relu(x)

                channel = channel * 2

            for i in range(2) :
                x = conv(x, channel, kernel=4, stride=2, pad=1, pad_type='reflect', scope='down_conv_'+str(i))
                x = relu(x)

            x = adaptive_avg_pooling(x) # global average pooling
            x = conv(x, self.style_dim, kernel=1, stride=1, scope='SE_logit')

            return x

    def Content_Encoder(self, x, reuse=False, scope='content_encoder'):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse) :
            x = conv(x, channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv_0')
            x = instance_norm(x, scope='ins_0')
            x = relu(x)

            for i in range(self.n_downsample) :
                x = conv(x, channel*2, kernel=4, stride=2, pad=1, pad_type='reflect', scope='conv_'+str(i+1))
                x = instance_norm(x, scope='ins_'+str(i+1))
                x = relu(x)

                channel = channel * 2

            for i in range(self.n_res) :
                x = resblock(x, channel, scope='resblock_'+str(i))

            return x

    def generator(self, contents, style, reuse=False, scope="decoder"):
        channel = self.mlp_dim
        with tf.variable_scope(scope, reuse=reuse) :
            mu, var = self.MLP(style)
            x = contents

            for i in range(self.n_res) :
                idx = 2 * i
                x = adaptive_resblock(x, channel, mu[idx], var[idx], mu[idx + 1], var[idx + 1], scope='adaptive_resblock_'+str(i))

            for i in range(self.n_upsample) :
                # # IN removes the original feature mean and variance that represent important style information
                x = up_sample(x, scale_factor=2)
                x = conv(x, channel//2, kernel=5, stride=1, pad=2, pad_type='reflect', scope='conv_'+str(i))
                x = layer_norm(x, scope='layer_norm_'+str(i))
                x = relu(x)

                channel = channel // 2

            x = conv(x, channels=self.img_ch, kernel=7, stride=1, pad=3, pad_type='reflect', scope='G_logit')
            x = tanh(x)

            return x

    def MLP(self, style, scope='MLP'):
        channel = self.mlp_dim
        with tf.variable_scope(scope) :
            x = style

            for i in range(2):
                x = fully_connected(x, channel, scope='FC_' + str(i))
                x = relu(x)

            mu_list = []
            var_list = []

            for i in range(self.n_res * 2):
                mu = fully_connected(x, channel, scope='FC_mu_' + str(i))
                var = fully_connected(x, channel, scope='FC_var_' + str(i))

                mu = tf.reshape(mu, shape=[-1, 1, 1, channel])
                var = tf.reshape(var, shape=[-1, 1, 1, channel])

                mu_list.append(mu)
                var_list.append(var)

            return mu_list, var_list

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x_init, reuse=False, scope="discriminator"):
        D_logit = []
        with tf.variable_scope(scope, reuse=reuse) :
            for scale in range(self.n_scale) :
                channel = self.ch
                x = conv(x_init, channel, kernel=4, stride=2, pad=1, pad_type='reflect', scope='ms_' + str(scale) + 'conv_0')
                x = lrelu(x, 0.2)

                for i in range(1, self.n_dis):
                    x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', scope='ms_' + str(scale) +'conv_' + str(i))
                    x = lrelu(x, 0.2)

                    channel = channel * 2

                x = conv(x, channels=1, kernel=1, stride=1, scope='ms_' + str(scale) + 'D_logit')
                D_logit.append(x)

                x_init = down_sample(x_init)

            return D_logit

    ##################################################################################
    # Model
    ##################################################################################

    def Encoder_A(self, x_A, reuse=False):
        style_A = self.Style_Encoder(x_A, reuse=reuse, scope='style_encoder_A')
        content_A = self.Content_Encoder(x_A, reuse=reuse, scope='content_encoder_A')

        return content_A, style_A

    def Encoder_B(self, x_B, reuse=False):
        style_B = self.Style_Encoder(x_B, reuse=reuse, scope='style_encoder_B')
        content_B = self.Content_Encoder(x_B, reuse=reuse, scope='content_encoder_B')

        return content_B, style_B

    # Instance encoder
    def Encoder_a(self, x_a, reuse=True):
        style_a = self.Style_Encoder(x_a, reuse=reuse, scope='style_encoder_A')
        content_a = self.Content_Encoder(x_a, reuse=reuse, scope='content_encoder_A')

        return content_a, style_a

    def Encoder_b(self, x_b, reuse=True):
        style_b = self.Style_Encoder(x_b, reuse=reuse, scope='style_encoder_B')
        content_b = self.Content_Encoder(x_b, reuse=reuse, scope='content_encoder_B')

        return content_b, style_b

    # global decoder
    def Decoder_A(self, content_B, style_A, reuse=False):
        x_ba = self.generator(contents=content_B, style=style_A, reuse=reuse, scope='decoder_A')

        return x_ba

    def Decoder_B(self, content_A, style_B, reuse=False):
        x_ab = self.generator(contents=content_A, style=style_B, reuse=reuse, scope='decoder_B')

        return x_ab

    # instance decoder
    def Decoder_a(self, content_b, style_a, reuse=False):
        x_ab = self.generator(contents=content_b, style=style_a, reuse=reuse, scope='decoder_a')
        return x_ab

    def Decoder_b(self, content_a, style_b, reuse=False):
        x_ba = self.generator(contents=content_a, style=style_b, reuse=reuse, scope='decoder_b')
        return x_ba

    def discriminate_real(self, x_A, x_B):
        real_A_logit = self.discriminator(x_A, scope="discriminator_A")
        real_B_logit = self.discriminator(x_B, scope="discriminator_B")

        return real_A_logit, real_B_logit

    def discriminate_fake(self, x_ba, x_ab):
        fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
        fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

        return fake_A_logit, fake_B_logit

    def build_model(self):
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Input Image"""
        Image_Data_Class = ImageData(self.img_h, self.img_w, self.img_ch, self.augment_flag)

        self.dataset_num = self.dataset()
        # os.system("pause")

        trainA = np.load(self.dataset_path_trainA)
        trainB = np.load(self.dataset_path_trainB)
        print()
        print('##### test info ####')
        print(type(trainA[0]))
        for key, value in trainA[0]:
            print(key, value)

        trainA = tf.data.Dataset.from_tensor_slices(trainA[:2])
        trainB = tf.data.Dataset.from_tensor_slices(trainB[:2])

        trainA = trainA.prefetch(self.batch_size).\
            shuffle(self.dataset_num).\
            map(Image_Data_Class.processing,num_parallel_calls=8).\
            apply(batch_and_drop_remainder(self.batch_size)).repeat()

        trainB = trainB.prefetch(self.batch_size).\
            shuffle(self.dataset_num).\
            map(Image_Data_Class.processing,num_parallel_calls=8).\
            apply(batch_and_drop_remainder(self.batch_size)).repeat()

        # trainA = trainA.map(Image_Data_Class.processing)
        # trainB = trainB.map(Image_Data_Class.processing)

        trainA_iterator = trainA.make_one_shot_iterator()
        trainB_iterator = trainB.make_one_shot_iterator()

        self.domain_A_all = trainA_iterator.get_next()
        self.domain_B_all = trainB_iterator.get_next()

        self.domain_A = self.domain_A_all['global']
        self.domain_a = self.domain_A_all['instances']
        self.domain_a_bg = self.domain_A_all['background']

        self.domain_B = self.domain_B_all['global']
        self.domain_b = self.domain_B_all['instances']
        self.domain_b_bg = self.domain_B_all['background']


        """ Define Encoder, Generator, Discriminator """
        print()
        print(" Define Encoder, Generator, Discriminator ")
        self.style_a = tf.placeholder(tf.float32, shape=[self.batch_size, 1, 1, self.style_dim], name='style_a')
        self.style_b = tf.placeholder(tf.float32, shape=[self.batch_size, 1, 1, self.style_dim], name='style_b')

        self.style_ao = tf.placeholder(tf.float32, shape=[self.batch_size, 1, 1, self.style_dim], name='style_ao')
        self.style_bo = tf.placeholder(tf.float32, shape=[self.batch_size, 1, 1, self.style_dim], name='style_bo')


        # encode (global)
        content_a, style_a_prime = self.Encoder_A(self.domain_A)
        content_b, style_b_prime = self.Encoder_B(self.domain_B)

        # encode (background)
        c_a_bg, s_a_bg_prime = self.Encoder_A(self.domain_a_bg)
        c_b_bg, s_b_bg_prime = self.Encoder_A(self.domain_b_bg)

        # instance encode
        c_a, s_a_prime = self.Encoder_a(self.domain_a)
        c_b, s_b_prime = self.Encoder_b(self.domain_b)

        # decode (within domain)
        # global
        x_aa = self.Decoder_A(content_B=content_a, style_A=style_a_prime)
        x_bb = self.Decoder_B(content_A=content_b, style_B=style_b_prime)
        # instance
        x_aa_o = self.Decoder_a(content_b=c_a, style_a=s_a_prime)
        x_bb_o = self.Decoder_b(content_a=c_b, style_b=s_b_prime)

        # decode (cross domain)
        # among images
        x_ba = self.Decoder_A(content_B=content_b, style_A=self.style_a, reuse=True)
        x_ab = self.Decoder_B(content_A=content_a, style_B=self.style_b, reuse=True)
        # among instances
        x_ba_o = self.Decoder_a(content_b=c_b, style_a=self.style_ao, reuse=True)
        x_ab_o = self.Decoder_b(content_a=c_a, style_b=self.style_bo, reuse=True)

        # decode (cross granularity)
        x_Aa = self.Decoder_a(content_b=c_a, style_a=style_a_prime, reuse=True)
        x_Bb = self.Decoder_b(content_a=c_b, style_b=style_b_prime, reuse=True)
        x_Aa_bg = self.Decoder_a(content_b=c_a, style_a=s_a_bg_prime, reuse=True)
        x_Bb_bg = self.Decoder_b(content_a=c_b, style_b=s_b_bg_prime, reuse=True)


        # encode again
        # cross domain (global)
        content_b_, style_a_ = self.Encoder_A(x_ba, reuse=True)
        content_a_, style_b_ = self.Encoder_B(x_ab, reuse=True)
        # cross domain (instance)
        c_a_, s_a_ = self.Encoder_a(x_ab_o, reuse=True)
        c_b_, s_b_ = self.Encoder_a(x_ba_o, reuse=True)
        # cross granularity (instance & global)
        c_a_o, s_a_g = self.Encoder_a(x_Aa, reuse=True)
        c_b_o, s_b_g = self.Encoder_b(x_Bb, reuse=True)
        # cross granularity (instance & background)
        c_a_obg, s_a_bg = self.Encoder_a(x_Aa_bg, reuse=True)
        c_b_obg, s_b_bg = self.Encoder_b(x_Bb_bg, reuse=True)


        # decode again (if needed)
        if self.recon_x_cyc_w > 0 :
            x_aba = self.Decoder_A(content_B=content_a_, style_A=style_a_prime, reuse=True)
            x_bab = self.Decoder_B(content_A=content_b_, style_B=style_b_prime, reuse=True)

            x_aba_o = self.Decoder_a(content_b=c_a_, style_a=s_a_prime)
            x_bab_o = self.Decoder_b(content_a=c_b_, style_b=s_b_prime)

            x_aba_og = self.Decoder_a(content_b=c_a_o, style_a=s_a_prime)
            x_bab_og = self.Decoder_b(content_a=c_b_o, style_b=s_b_prime)

            x_aba_ob = self.Decoder_a(content_b=c_a_o, style_a=s_a_bg_prime)
            x_bab_ob = self.Decoder_b(content_a=c_b_o, style_b=s_b_bg_prime)

            cyc_recon_A = L1_loss(x_aba, self.domain_A)
            cyc_recon_B = L1_loss(x_bab, self.domain_B)

            cyc_recon_a = L1_loss(x_aba_o, self.domain_a)
            cyc_recon_b = L1_loss(x_bab_o, self.domain_b)

            cyc_recon_ag = L1_loss(x_aba_og, self.domain_a)
            cyc_recon_bg = L1_loss(x_bab_og, self.domain_b)

            cyc_recon_ab = L1_loss(x_aba_ob, self.domain_a)
            cyc_recon_bb = L1_loss(x_bab_ob, self.domain_b)

        else :
            cyc_recon_A = 0.0
            cyc_recon_B = 0.0
            cyc_recon_a = 0.0
            cyc_recon_b = 0.0
            cyc_recon_ag = 0.0
            cyc_recon_bg = 0.0
            cyc_recon_ab = 0.0
            cyc_recon_bb = 0.0

        real_A_logit, real_B_logit = self.discriminate_real(self.domain_A, self.domain_B)
        fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
        # instance (cross domain)
        real_a_logit, real_b_logit = self.discriminate_real(self.domain_a, self.domain_b)
        fake_a_logit, fake_b_logit = self.discriminate_fake(x_ba_o, x_ab_o)
        # instance (cross granularity: global)
        # real_ag_logit, real_bg_logit = self.discriminate_real(self)
        fake_ag_logit, fake_bg_logit = self.discriminate_fake(x_Aa, x_Bb)
        # instance (cross granularity: background)
        fake_abg_logit, fake_bbg_logit = self.discriminate_fake(x_Aa_bg, x_Bb_bg)

        """ Define Loss """
        print(" Define Loss ")
        # Generator loss
        G_ad_loss_a = generator_loss(self.gan_type, fake_A_logit)
        G_ad_loss_b = generator_loss(self.gan_type, fake_B_logit)

        G_ad_loss_a_o = generator_loss(self.gan_type, fake_a_logit)
        G_ad_loss_b_o = generator_loss(self.gan_type, fake_b_logit)

        # instance & global
        G_ad_loss_ag = generator_loss(self.gan_type, fake_ag_logit)
        G_ad_loss_bg = generator_loss(self.gan_type, fake_bg_logit)
        # instance & background
        G_ad_loss_abg = generator_loss(self.gan_type, fake_abg_logit)
        G_ad_loss_bbg = generator_loss(self.gan_type, fake_bbg_logit)

        G_ad_loss_ao = G_ad_loss_a_o + G_ad_loss_ag + G_ad_loss_abg
        G_ad_loss_bo = G_ad_loss_b_o + G_ad_loss_bg + G_ad_loss_bbg

        D_ad_loss_a = discriminator_loss(self.gan_type, real_A_logit, fake_A_logit)
        D_ad_loss_b = discriminator_loss(self.gan_type, real_B_logit, fake_B_logit)


        D_ad_loss_ao = discriminator_loss(self.gan_type, real_a_logit, fake_a_logit) + \
                       discriminator_loss(self.gan_type, real_a_logit, fake_ag_logit)
        D_ad_loss_bo = discriminator_loss(self.gan_type, real_b_logit, fake_b_logit) + \
                       discriminator_loss(self.gan_type, real_a_logit, fake_bg_logit)
        # D_ad_loss_ag = discriminator_loss(self.gan_type, real_a_logit, fake_ag_logit)
        # D_ad_loss_bg = discriminator_loss(self.gan_type, real_a_logit, fake_bg_logit)

        # Reconstarction loss
        # global
        recon_A = L1_loss(x_aa, self.domain_A) # reconstruction
        recon_B = L1_loss(x_bb, self.domain_B) # reconstruction
        # instance
        recon_a = L1_loss(x_aa_o, self.domain_a)
        recon_b = L1_loss(x_bb_o, self.domain_b)

        # The style reconstruction loss encourages
        # diverse outputs given different style codes
        # global
        recon_style_A = L1_loss(style_a_, self.style_a) + L1_loss(s_a_g, self.style_a)
        recon_style_B = L1_loss(style_b_, self.style_b) + L1_loss(s_b_g, self.style_b)

        # instance
        recon_s_a = L1_loss(s_a_, self.style_ao)
        recon_s_b = L1_loss(s_b_, self.style_bo)


        # The content reconstruction loss encourages
        # the translated image to preserve semantic content of the input image
        recon_content_A = L1_loss(content_a_, content_a)
        recon_content_B = L1_loss(content_b_, content_b)

        # instance
        # recon from cross domain
        recon_c_a = L1_loss(c_a_, c_a) + L1_loss(c_a_o, c_a) + L1_loss(c_a_obg, c_a)
        recon_c_b = L1_loss(c_b_, c_b) + L1_loss(c_b_o, c_b) + L1_loss(c_b_obg, c_b)


        Generator_A_loss = self.gan_w * G_ad_loss_a + \
                           self.recon_x_w * recon_A + \
                           self.recon_s_w * recon_style_A + \
                           self.recon_c_w * recon_content_A + \
                           self.recon_x_cyc_w * cyc_recon_A


        Generator_B_loss = self.gan_w * G_ad_loss_b + \
                           self.recon_x_w * recon_B + \
                           self.recon_s_w * recon_style_B + \
                           self.recon_c_w * recon_content_B + \
                           self.recon_x_cyc_w * cyc_recon_B

        Generator_a_loss = self.gan_o_w * G_ad_loss_ao + \
                           self.recon_o_w * recon_a + \
                           self.recon_o_s_w * recon_s_a + \
                           self.recon_o_c_w * recon_c_a + \
                           self.recon_o_cyc_w * cyc_recon_a + \
                           self.recon_o_cyc_w * cyc_recon_a + \
                           self.recon_o_cyc_w * cyc_recon_ab + \
                           self.recon_o_cyc_w * cyc_recon_ag

        Generator_b_loss = self.gan_o_w * G_ad_loss_bo + \
                           self.recon_o_w * recon_b + \
                           self.recon_o_s_w * recon_s_b + \
                           self.recon_o_c_w * recon_c_b + \
                           self.recon_o_cyc_w * cyc_recon_b + \
                           self.recon_o_cyc_w * cyc_recon_b + \
                           self.recon_o_cyc_w * cyc_recon_bb + \
                           self.recon_o_cyc_w * cyc_recon_bg

        Discriminator_A_loss = self.gan_w * D_ad_loss_a + self.gan_o_w * D_ad_loss_ao
        Discriminator_B_loss = self.gan_w * D_ad_loss_b + self.gan_o_w * D_ad_loss_bo

        self.Generator_loss = Generator_A_loss + Generator_B_loss + \
                              Generator_a_loss + Generator_b_loss + \
                              regularization_loss('encoder') + regularization_loss('decoder')
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss + regularization_loss('discriminator')

        """ Training """
        print("define training")
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'decoder' in var.name or 'encoder' in var.name]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.G_a_loss = tf.summary.scalar("G_a_loss", Generator_a_loss)
        self.G_b_loss = tf.summary.scalar("G_b_loss", Generator_b_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.G_A_loss, self.G_a_loss, self.G_B_loss, self.G_B_loss, self.all_G_loss])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        """ Image """
        self.fake_A = x_ba
        self.fake_B = x_ab
        self.fake_a = x_ba_o
        self.fake_b = x_ab_o

        self.real_A = self.domain_A
        self.real_B = self.domain_B
        self.real_a = self.domain_a
        self.real_b = self.domain_b

        """ Test """
        self.test_image = tf.placeholder(tf.float32, [1, self.img_h, self.img_w, self.img_ch], name='test_image')
        self.test_style = tf.placeholder(tf.float32, [1, 1, 1, self.style_dim], name='test_style')
        # 120*120
        self.test_instance = tf.placeholder(tf.float32, [1, self.istn_w, self.istn_w, self.img_ch], name='test_instance')
        self.test_local_style = tf.placeholder(tf.float32, [1, 1, 1, self.style_dim], name='test_instance_style')

        test_content_a, _ = self.Encoder_A(self.test_image, reuse=True)
        test_content_b, _ = self.Encoder_B(self.test_image, reuse=True)

        self.test_fake_A = self.Decoder_A(content_B=test_content_b, style_A=self.test_style, reuse=True)
        self.test_fake_B = self.Decoder_B(content_A=test_content_a, style_B=self.test_style, reuse=True)

        test_constent_oa, _ = self.Encoder_a(self.test_instance, reuse=True)
        test_constent_ob, _ = self.Encoder_b(self.test_instance, reuse=True)

        self.test_fake_oa = self.Decoder_a(content_b=test_constent_oa, style_a=self.test_local_style, reuse=True)
        self.test_fake_ob = self.Decoder_b(content_a=test_constent_ob, style_b=self.test_local_style, reuse=True)

        """ Guided Image Translation """
        print("define translation")
        self.content_image = tf.placeholder(tf.float32, [1, self.img_h, self.img_w, self.img_ch], name='content_image')
        self.style_image = tf.placeholder(tf.float32, [1, self.img_h, self.img_w, self.img_ch], name='guide_style_image')

        self.content_o = tf.placeholder(tf.float32, [1, self.inst_w, self.inst_w, self.img_ch], name='content_instance')
        self.style_o = tf.placeholder(tf.float32, [1, self.inst_w, self.inst_w, self.img_ch], name='guide_style_instance')

        if self.direction == 'a2b' :
            guide_content_A, guide_style_A = self.Encoder_A(self.content_image, reuse=True)
            guide_content_B, guide_style_B = self.Encoder_B(self.style_image, reuse=True)
            guide_content_oa, guide_style_oa = self.Encoder_a(self.content_o, reuse=True)
            guide_content_ob, guide_style_ob = self.Encoder_b(self.style_o, reuse=True)

        else :
            guide_content_B, guide_style_B = self.Encoder_B(self.content_image, reuse=True)
            guide_content_A, guide_style_A = self.Encoder_A(self.style_image, reuse=True)
            guide_content_oa, guide_style_oa = self.Encoder_a(self.style_o, reuse=True)
            guide_content_ob, guide_style_ob = self.Encoder_b(self.content_o, reuse=True)

        self.guide_fake_A = self.Decoder_A(content_B=guide_content_B, style_A=guide_style_A, reuse=True)
        self.guide_fake_B = self.Decoder_B(content_A=guide_content_A, style_B=guide_style_B, reuse=True)
        self.guide_fake_a = self.Decoder_a(content_b=guide_content_ob, style_a=guide_style_oa, reuse=True)
        self.guide_fake_b = self.Decoder_b(content_a=guide_content_oa, style_b=guide_style_ob, reuse=True)

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            lr = self.init_lr * pow(0.5, epoch)

            for idx in range(start_batch_id, self.iteration):
                style_a = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, 1, 1, self.style_dim])
                style_b = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, 1, 1, self.style_dim])
                style_oa = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, 1, 1, self.style_dim])
                style_ob = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, 1, 1, self.style_dim])

                train_feed_dict = {
                    self.style_a : style_a,
                    self.style_b : style_b,
                    self.style_ao : style_oa,
                    self.style_bo : style_ob,
                    self.lr : lr
                }

                # Update D
                _, d_loss, summary_str = self.sess.run([self.D_optim, self.Discriminator_loss, self.D_loss], feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # Update G
                batch_A_images, batch_B_images, batch_A_instances, batch_B_instances, fake_A, fake_B, fake_a, fake_b, _, g_loss, summary_str = \
                    self.sess.run([self.real_A, self.real_B, self.real_a, self.real_b, self.fake_A, self.fake_B, self.fake_a, self.fake_b, self.G_optim, self.Generator_loss, self.G_loss], feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%6d/%6d] time: %4.4f d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, d_loss, g_loss))

                if np.mod(idx+1, self.print_freq) == 0 :
                    save_images(batch_A_images, [self.batch_size, 1],
                                './{}/real_A_{:02d}_{:06d}.jpg'.format(self.sample_dir, epoch, idx+1))
                    # save_images(batch_B_images, [self.batch_size, 1],
                    #             './{}/real_B_{}_{:02d}_{:06d}.jpg'.format(self.sample_dir, gpu_id, epoch, idx+1))

                    # save_images(fake_A, [self.batch_size, 1],
                    #             './{}/fake_A_{}_{:02d}_{:06d}.jpg'.format(self.sample_dir, gpu_id, epoch, idx+1))
                    save_images(fake_B, [self.batch_size, 1],
                                './{}/fake_B_{:02d}_{:06d}.jpg'.format(self.sample_dir, epoch, idx+1))

                if np.mod(idx+1, self.save_freq) == 0 :
                    self.save(self.checkpoint_dir, counter)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model for final step
            self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}_{}_{}".format(self.model_name, self.dataset_name, self.gan_type)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()
        test_A_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testA'))
        test_B_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testB'))

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file  in test_A_files : # A -> B
            print('Processing A image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, size_h=self.img_h, size_w=self.img_w))
            file_name = os.path.basename(sample_file).split(".")[0]
            file_extension = os.path.basename(sample_file).split(".")[1]

            for i in range(self.num_style) :
                test_style = np.random.normal(loc=0.0, scale=1.0, size=[1, 1, 1, self.style_dim])
                image_path = os.path.join(self.result_dir, '{}_style{}.{}'.format(file_name, i, file_extension))

                fake_img = self.sess.run(self.test_fake_B, feed_dict = {self.test_image : sample_image, self.test_style : test_style})
                save_images(fake_img, [1, 1], image_path)

                index.write("<td>%s</td>" % os.path.basename(image_path))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                    '../..' + os.path.sep + sample_file), self.img_w, self.img_h))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                    '../..' + os.path.sep + image_path), self.img_w, self.img_h))
                index.write("</tr>")

        for sample_file  in test_B_files : # B -> A
            print('Processing B image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, size_h=self.img_h, size_w=self.img_w))
            file_name = os.path.basename(sample_file).split(".")[0]
            file_extension = os.path.basename(sample_file).split(".")[1]

            for i in range(self.num_style):
                test_style = np.random.normal(loc=0.0, scale=1.0, size=[1, 1, 1, self.style_dim])
                image_path = os.path.join(self.result_dir, '{}_style{}.{}'.format(file_name, i, file_extension))

                fake_img = self.sess.run(self.test_fake_A, feed_dict={self.test_image: sample_image, self.test_style: test_style})
                save_images(fake_img, [1, 1], image_path)

                index.write("<td>%s</td>" % os.path.basename(image_path))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                        '../..' + os.path.sep + sample_file), self.img_w, self.img_h))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                        '../..' + os.path.sep + image_path), self.img_w, self.img_h))
                index.write("</tr>")
        index.close()

    def style_guide_test(self):
        tf.global_variables_initializer().run()
        test_A_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testA'))
        test_B_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testB'))

        style_file = np.asarray(load_test_data(self.guide_img, size_h=self.img_h, size_w=self.img_w))

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir, 'guide')
        check_folder(self.result_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        if self.direction == 'a2b' :
            for sample_file in test_A_files:  # A -> B
                print('Processing A image: ' + sample_file)
                sample_image = np.asarray(load_test_data(sample_file, size_h=self.img_h, size_w=self.img_w))
                image_path = os.path.join(self.result_dir, '{}'.format(os.path.basename(sample_file)))

                fake_img = self.sess.run(self.guide_fake_B, feed_dict={self.content_image: sample_image, self.style_image : style_file})
                save_images(fake_img, [1, 1], image_path)

                index.write("<td>%s</td>" % os.path.basename(image_path))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                        '../../..' + os.path.sep + sample_file), self.img_w, self.img_h))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                        '../../..' + os.path.sep + image_path), self.img_w, self.img_h))
                index.write("</tr>")

        else :
            for sample_file in test_B_files:  # B -> A
                print('Processing B image: ' + sample_file)
                sample_image = np.asarray(load_test_data(sample_file, size_h=self.img_h, size_w=self.img_w))
                image_path = os.path.join(self.result_dir, '{}'.format(os.path.basename(sample_file)))

                fake_img = self.sess.run(self.guide_fake_A, feed_dict={self.content_image: sample_image, self.style_image : style_file})
                save_images(fake_img, [1, 1], image_path)

                index.write("<td>%s</td>" % os.path.basename(image_path))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                        '../../..' + os.path.sep + sample_file), self.img_w, self.img_h))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                        '../../..' + os.path.sep + image_path), self.img_w, self.img_h))
                index.write("</tr>")
        index.close()



    def dataset(self):
        print("_"*20)
        print()
        print("start to process data")
        print('##### test info #####')
        if os.path.exists(self.dataset_path_trainA) and os.path.exists(self.dataset_path_trainB) and os.path.exists(self.dataset_path_testA) and os.path.exists(self.dataset_path_testB):
            trainA = pickle.load(open(self.dataset_path_trainA, 'rb'))
            trainB = pickle.load(open(self.dataset_path_trainB, 'rb'))
        else:
            if not os.path.exists(self.dataset_before_split):
                folder_name = ['cloudy', 'rainy', 'sunny', 'night']
                all_images = dict()
                weather_list = os.listdir(self.data_folder)
                for weather in weather_list:
                    if weather not in folder_name:
                        continue

                    weather_dir = os.path.join(self.data_folder, weather)
                    if os.path.isdir(weather_dir):
                        get_files(weather_dir, all_images)

                pickle.dump(all_images, open(self.dataset_before_split, 'wb'))
                # pickle.dump(all_images, open('/home/user/share/dataset/data/all_data.pkl', 'wb'))
            else:
                all_images = pickle.load(open(self.dataset_before_split, 'rb'))
                # all_images = pickle.load(open('/home/user/share/dataset/data/all_data.pkl', 'rb'))

            print('dividing data into part A & part B')
            print('all images : ', type(all_images), len(all_images))
            new_data = []
            for key, value in all_images.items():
                if len(value['instance']) > 0:
                    temp = value['instance']
                    value['instance'] = temp
                    new_data.append(value)
            data_num = len(new_data) // 2
            random.shuffle(new_data)
            print('new data', len(new_data))
            trainA = []
            trainB = []
            count = 0
            for i in range(data_num * 2):
                if i < data_num:
                    trainA.append(new_data[i])
                else:
                    trainB.append(new_data[i])
            # for key, value in all_images.items():
            #     if count < data_num:
            #         trainA.append(all_images[key])
            #     else:
            #         trainB.append(all_images[key])
            #     count += 1

            # print('trainA : ', type(trainA[0]))
            # for key, value in trainA[0].items():
            #     print(key, value)
            print('##### data test end ######')
            # import pdb; pdb.set_trace()
            # split data
            trainA, trainB, testA, testB = train_test_split(trainA, trainB, test_size=0.2, random_state=0)
            pickle.dump(trainA, open('/home/user/share/dataset/data/trainA.pkl', 'wb'))
            pickle.dump(trainB, open('/home/user/share/dataset/data/trainB.pkl', 'wb'))
            pickle.dump(testA, open('/home/user/share/dataset/data/testA.pkl', 'wb'))
            pickle.dump(testB, open('/home/user/share/dataset/data/testB.pkl', 'wb'))

            # np.save(self.dataset_path_trainA, trainA)
            # np.save(self.dataset_path_trainB, trainB)
            # np.save(self.dataset_path_testA, testA)
            # np.save(self.dataset_path_testB, testB)

        print("data ready")
        print()

        return max(len(trainA), len(trainB))
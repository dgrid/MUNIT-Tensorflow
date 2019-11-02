from MUNIT import MUNIT
from INIT import INIT
import argparse
from utils import *

"""parsing and configuration"""
# def parse_args():
#     desc = "Tensorflow implementation of MUNIT"
#     parser = argparse.ArgumentParser(description=desc)
#     parser.add_argument('--phase', type=str, default='train', help='train or test or guide')
#     parser.add_argument('--model', type=str, default='MUNIT', help='which model to use')
#     parser.add_argument('--dataset', type=str, default='summer2winter', help='dataset_name')
#     parser.add_argument('--augment_flag', type=bool, default=False, help='Image augmentation use or not')
#
#     parser.add_argument('--epoch', type=int, default=10, help='The number of epochs to run')
#     parser.add_argument('--iteration', type=int, default=100000, help='The number of training iterations')
#     parser.add_argument('--batch_size', type=int, default=1, help='The batch size')
#     parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freq')
#     parser.add_argument('--save_freq', type=int, default=1000, help='The number of ckpt_save_freq')
#     parser.add_argument('--num_style', type=int, default=3, help='number of styles to sample')
#     parser.add_argument('--direction', type=str, default='a2b', help='direction of style guided image translation')
#     parser.add_argument('--guide_img', type=str, default='guide.jpg', help='Style guided image translation')
#
#     parser.add_argument('--gan_type', type=str, default='lsgan', help='GAN loss type [gan / lsgan]')
#
#     parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
#     parser.add_argument('--gan_w', type=float, default=1.0, help='weight of adversarial loss')
#     parser.add_argument('--recon_x_w', type=float, default=10.0, help='weight of image reconstruction loss')
#     parser.add_argument('--recon_s_w', type=float, default=1.0, help='weight of style reconstruction loss')
#     parser.add_argument('--recon_c_w', type=float, default=1.0, help='weight of content reconstruction loss')
#     parser.add_argument('--recon_x_cyc_w', type=float, default=0.0, help='weight of explicit style augmented cycle consistency loss')
#
#     parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
#     parser.add_argument('--style_dim', type=int, default=8, help='length of style code')
#     parser.add_argument('--n_sample', type=int, default=2, help='number of sampling layers in content encoder')
#     parser.add_argument('--n_res', type=int, default=4, help='number of residual blocks in content encoder/decoder')
#
#     parser.add_argument('--n_dis', type=int, default=4, help='number of discriminator layer')
#     parser.add_argument('--n_scale', type=int, default=3, help='number of scales')
#
#     parser.add_argument('--img_h', type=int, default=256, help='The size of image hegiht')
#     parser.add_argument('--img_w', type=int, default=256, help='The size of image width')
#     parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
#
#     parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
#                         help='Directory name to save the checkpoints')
#     parser.add_argument('--result_dir', type=str, default='results',
#                         help='Directory name to save the generated images')
#     parser.add_argument('--log_dir', type=str, default='logs',
#                         help='Directory name to save training logs')
#     parser.add_argument('--sample_dir', type=str, default='samples',
#                         help='Directory name to save the samples on training')
#
#     return check_args(parser.parse_args())

def parse_args():
    desc = "Tensorflow implementation of INIT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test or guide')
    parser.add_argument('--model', type=str, default='INIT', help='which model to use')
    # parser.add_argument('--dataset', type=str, default='summer2winter', help='dataset_name')
    parser.add_argument('--augment_flag', type=bool, default=False, help='Image augmentation use or not')

    # parser.add_argument('--epoch', type=int, default=10, help='The number of epochs to run')
    # parser.add_argument('--iteration', type=int, default=100000, help='The number of training iterations')
    # parser.add_argument('--batch_size', type=int, default=1, help='The batch size')
    # parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freq')
    # parser.add_argument('--save_freq', type=int, default=1000, help='The number of ckpt_save_freq')
    # parser.add_argument('--num_style', type=int, default=3, help='number of styles to sample')
    # parser.add_argument('--direction', type=str, default='a2b', help='direction of style guided image translation')
    # parser.add_argument('--guide_img', type=str, default='guide.jpg', help='Style guided image translation')

    parser.add_argument('--epoch', type=int, default=10, help='The number of epochs to run')
    parser.add_argument('--iteration', type=int, default=10, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size')
    parser.add_argument('--print_freq', type=int, default=1, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=1, help='The number of ckpt_save_freq')
    parser.add_argument('--num_style', type=int, default=3, help='number of styles to sample')
    parser.add_argument('--direction', type=str, default='a2b', help='direction of style guided image translation')
    parser.add_argument('--guide_img', type=str, default='guide.jpg', help='Style guided image translation')

    parser.add_argument('--gan_type', type=str, default='lsgan', help='GAN loss type [gan / lsgan]')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--gan_w', type=float, default=1.0, help='weight of adversarial loss')
    parser.add_argument('--recon_x_w', type=float, default=10.0, help='weight of image reconstruction loss')
    parser.add_argument('--recon_s_w', type=float, default=1.0, help='weight of style reconstruction loss')
    parser.add_argument('--recon_c_w', type=float, default=1.0, help='weight of content reconstruction loss')
    parser.add_argument('--recon_x_cyc_w', type=float, default=0.0,
                        help='weight of explicit style augmented cycle consistency loss')

    parser.add_argument('--gan_o_w', type=float, default=1.0, help='weight of adversarial loss')
    parser.add_argument('--recon_o_w', type=float, default=10.0, help='weight of image reconstruction loss-instance')
    parser.add_argument('--recon_o_s_w', type=float, default=1.0, help='weight of style reconstruction loss-instance')
    parser.add_argument('--recon_o_c_w', type=float, default=1.0, help='weight of content reconstruction loss-instance')
    parser.add_argument('--recon_o_cyc_w', type=float, default=0.0,
                        help='weight of explicit style augmented cycle consistency loss-instance')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--style_dim', type=int, default=8, help='length of style code')
    parser.add_argument('--n_sample', type=int, default=2, help='number of sampling layers in content encoder')
    parser.add_argument('--n_res', type=int, default=4, help='number of residual blocks in content encoder/decoder')

    parser.add_argument('--n_dis', type=int, default=4, help='number of discriminator layer')
    parser.add_argument('--n_scale', type=int, default=3, help='number of scales')

    parser.add_argument('--img_h', type=int, default=256, help='The size of image hegiht')
    parser.add_argument('--img_w', type=int, default=256, help='The size of image width')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--inst_w', type=int, default=120, help='The size of instance')

    parser.add_argument('--checkpoint_dir', type=str, default='INIT_checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='INIT_results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='INIT_logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='INIT_samples',
                        help='Directory name to save the samples on training')

    parser.add_argument('--dataset', type=str, default='INIT',
                        help='Directory name to save the samples on training')
    parser.add_argument('--data_folder', type=str, default='/home/user/share/dataset', help='dataset folder')
    parser.add_argument('--debug', action='store_true', help='use random dataset for debugging')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""main"""
def main():
    # parse arguments
    print("read args")
    args = parse_args()
    if args is None:
      exit()
      print()

    # allocate dynamically
    os.environ["CUDA_VISIBLE_DEVICES"] = '2' #use GPU with ID=0
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
    config.gpu_options.allow_growth = True

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if args.model == 'MUNIT':
            gan = MUNIT(sess, args)
        else:
            gan = INIT(sess, args)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        if args.phase == 'train' :
            # launch the graph in a session
            gan.train()
            print(" [*] Training finished!")

        if args.phase == 'test' :
            gan.test()
            print(" [*] Test finished!")

        if args.phase == 'guide' :
            gan.style_guide_test()
            print(" [*] Guide finished!")

if __name__ == '__main__':
    main()

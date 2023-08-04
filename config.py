import argparse

def input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./data/imgs', help='input RGB or Gray image path')
    parser.add_argument('--mask_dir', type=str, default='./data/masks', help='input mask path')
    parser.add_argument('--lr', type=float, default='0.0002', help='learning rate')
    parser.add_argument('--batch_size', type=int, default='8', help='batch_size in training')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--epoch", type=int, default=50, help="epoch in training")

    args = parser.parse_args()
    return args
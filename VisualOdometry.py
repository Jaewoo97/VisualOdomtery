
from __future__ import print_function
import argparse
import cv2 as cv
import features
import globals
import numpy as np
import csv
import open3d as o3d
import outputs
import save_figures
from progress.bar import Bar
import itertools
import glob
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import time
from submodule import *
from preprocess import *
from dataloader import listflowfile as lt
from dataloader.KITTILoader import *
from dataloader.KITTIloader2015 import *
import cv2 as cv



def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False), \
                         nn.BatchNorm2d(out_planes))

class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1,1,1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2,1,1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1,1,1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1,1,2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride = 1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output      = self.firstconv(x)
        output      = self.layer1(output)
        output_raw  = self.layer2(output)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)


        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=True)
        # output_branch1 = F.upsample(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=True)

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=True)

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=True)

        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature


class PSMNet(nn.Module):
    def __init__(self, maxdisp):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.feature_extraction = feature_extraction()
        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)
        self.dres3 = hourglass(32)
        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):
        refimg_fea = self.feature_extraction(left)
        targetimg_fea = self.feature_extraction(right)

        cost = Variable(
            torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1] * 2, self.maxdisp // 4, refimg_fea.size()[2],
                              refimg_fea.size()[3]).zero_()).cuda()

        for i in range(self.maxdisp // 4):
            if i > 0:
                cost[:, :refimg_fea.size()[1], i, :, i:] = refimg_fea[:, :, :, i:]
                cost[:, refimg_fea.size()[1]:, i, :, i:] = targetimg_fea[:, :, :, :-i]
            else:
                cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :, :] = targetimg_fea
        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2
        if self.training:
            cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')

            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)

            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)

        cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)

        pred3 = disparityregression(self.maxdisp)(pred3)

        if self.training:
            return pred1, pred2, pred3
        else:
            return pred3

if __name__=='__main__':
    max_disp = 192
    epochs = 1
    model = PSMNet(max_disp)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    print(model.feature_extraction)

def test_img(imgL, imgR):

    model.eval()

    imgL1 = imgL.cuda()
    imgR1 = imgR.cuda()

    with torch.no_grad():
        disp = model(imgL1, imgR1)

    disp = torch.squeeze(disp)
    pred_disp1 = disp.data.cpu().numpy()

    return pred_disp1

if __name__=='__main__':
    print('Start disparity calculation')
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    from collections import OrderedDict
    checkpoint = torch.load('pretrained_model_KITTI2015.tar')
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(**normal_mean_var)])

    for img_idx in range(70):  # Input number of frames

        #Input your path to stereo view images#
        left_image_path = 'valdata/left/Image0' + str(img_idx + 500) + '.jpg'
        right_image_path = 'valdata/right/Image0' + str(img_idx + 500) + '.jpg'
        ######################################
        imgL_o = Image.open(left_image_path).convert('RGB')
        imgR_o = Image.open(right_image_path).convert('RGB')

        imgL = infer_transform(imgL_o)
        imgR = infer_transform(imgR_o)

        # pad to width and hight to 16 times
        if imgL.shape[1] % 16 != 0:
            times = imgL.shape[1]//16
            top_pad = (times+1)*16 -imgL.shape[1]
        else:
            top_pad = 0

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16
            right_pad = (times+1)*16-imgL.shape[2]
        else:
            right_pad = 0

        imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
        imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

        stereo_sgbm = cv.StereoSGBM()
        start_time = time.time()
        pred_disp = test_img(imgL,imgR)
        if top_pad !=0 or right_pad != 0:
            img = pred_disp[top_pad:,:-right_pad]
        else:
            img = pred_disp

        print("saving disparity... itr: " + str(img_idx + 630))
        save_name = "disp0" + str(img_idx + 630) + ".csv"
        np.savetxt(save_name, img, delimiter=",")


# Initializing global variables
globals.initialize()
ransac_iterations = 15000
val = 1  # if not 1, test set
random_ransac = 0  # if not 1, try every combinations possible for ransac
manual_threshold_list = [0.05, 0.025] # if greater than 0, manual threshold is assigned
num_feats_list = [100]
DL_disparity = 1  # if 1, uses disparity calculated from deep learning
K = np.array([[318.951, -0.8, 322], [0, -318.951, 242], [0, 0, 1]])
baseline = 0.615  # meters
focal_length = 318  # pixels
globals.detector = features.SIFT()
globals.descriptor = features.SIFT()
matcher = 'BF'
descriptor = 'SIFT'
assign_mask = 1

# printing params
print_match_figures = 0
print_disparity = 0

# Iterate over sequence:
if val ==1:
    which_path = 'valdata/left/'
else:
    which_path = 'testdata/left/'

iterate_counter = len(glob.glob1(which_path, "*.jpg"))

# iterate over different parameters: num_features, threshold
for num_feats in num_feats_list:
    for manual_threshold in manual_threshold_list:
        output_arr = []
        inlier_values = []
        print('Starting iteration of num_feats: ', str(num_feats), ' threshold: ', str(manual_threshold))
        for iterate_idx in range(iterate_counter-1):
            b_text = "Iteration num: " + str(iterate_idx) + " out of " + str(iterate_counter)
            print(b_text, end="\r")
        # for iterate_idx in range(78,130):
            # Open and Convert the input image from BGR to GRAYSCALE
            if val == 1:
                prev_image_path = 'valdata/left/Image0' + str(iterate_idx + 500) + '.jpg'
                next_image_path = 'valdata/left/Image0' + str(iterate_idx + 501) + '.jpg'
            else:
                prev_image_path = 'testdata/left/Image0' + str(iterate_idx + 630) + '.jpg'
                next_image_path = 'testdata/left/Image0' + str(iterate_idx + 631) + '.jpg'

            image1 = cv.imread(filename=prev_image_path, flags=cv.IMREAD_GRAYSCALE)
            image2 = cv.imread(filename=next_image_path, flags=cv.IMREAD_GRAYSCALE)

            # Could not open or find the images
            if image1 is None or image2 is None:
                print('\nCould not open or find the images.')
                exit(0)

            # Find the keypoints and compute
            # the descriptors for input image
            globals.keypoints1, globals.descriptors1 = features.features(image1)

            # Find the keypoints and compute
            # the descriptors for training-set image
            globals.keypoints2, globals.descriptors2 = features.features(image2)

            # Matcher
            output, xy_matches, kp1, kp2 = features.matcher(image1 = image1,
                                      image2 = image2,
                                      keypoints1 = globals.keypoints1,
                                      keypoints2 = globals.keypoints2,
                                      descriptors1 = globals.descriptors1,
                                      descriptors2 = globals.descriptors2,
                                      matcher = matcher,
                                      descriptor = descriptor, num_features = num_feats, is_val=val, frame = iterate_idx, mask=assign_mask)

            if len(xy_matches)<num_feats:
                num_feats = len(xy_matches)
                print('frame: ', iterate_idx, ' only # of matches available: ', num_feats,'\n')
            # Save Figure Matcher
            if print_match_figures == 1:
                save_figures.saveMatcher(output = output,
                                         matcher = matcher,
                                         descriptor = descriptor,
                                         ransac = "no")

            # Print
            # print('Feature Description and Matching executed with success!')

            prev_xy = np.zeros([num_feats, 2])
            prev_xyz = np.zeros([num_feats, 3])
            next_xy = np.zeros([num_feats, 2])
            next_xyz = np.zeros([num_feats, 3])

            # Calculate depth

            if DL_disparity == 1:
                if val == 1:
                    prev_read_name = 'disp0' + str(iterate_idx + 630) + '.csv'
                    next_read_name = 'disp0' + str(iterate_idx + 631) + '.csv'
                else:
                    prev_read_name = 'disp0' + str(iterate_idx + 630) + '.csv'
                    next_read_name = 'disp0' + str(iterate_idx + 631) + '.csv'

                with open(prev_read_name, newline='') as csvfile:
                     foo = csv.reader(csvfile, delimiter=',', quotechar='|')
                     prev_disparity = np.array(list(foo))
                with open(next_read_name, newline='') as csvfile:
                     foo = csv.reader(csvfile, delimiter=',', quotechar='|')
                     next_disparity = np.array(list(foo))
            else:
                left_image_path = 'valdata/left/Image0' + str(iterate_idx + 500) + '.jpg'
                right_image_path = 'valdata/right/Image0' + str(iterate_idx + 500) + '.jpg'
                left_image_path1 = 'valdata/left/Image0' + str(iterate_idx + 501) + '.jpg'
                right_image_path1 = 'valdata/right/Image0' + str(iterate_idx + 501) + '.jpg'
                ImT1_L = cv.imread(left_image_path, 0)
                ImT1_R = cv.imread(right_image_path, 0)
                ImT2_L = cv.imread(left_image_path1, 0)
                ImT2_R = cv.imread(right_image_path1, 0)
                block = 31
                P1 = 160
                P2 = 510
                disparityEngine = cv.StereoSGBM_create(minDisparity=0, numDisparities=96, blockSize=block, P1=P1, P2=P2)
                ImT1_disparity = disparityEngine.compute(ImT1_L, ImT1_R).astype(np.float32)
                plt.title('disparity output')
                plt.imshow(ImT1_disparity)
                plt.show()

                cv.imwrite('disparity.png', ImT1_disparity)
                prev_disparity = np.divide(ImT1_disparity, 16.0)

                ImT2_disparity = disparityEngine.compute(ImT2_L, ImT2_R).astype(np.float32)
                next_disparity = np.divide(ImT2_disparity, 16.0)



            prev_depth = np.zeros([prev_disparity.shape[0], prev_disparity.shape[1]])
            next_depth = np.zeros([next_disparity.shape[0], next_disparity.shape[1]])
            for row_idx, row in enumerate(prev_disparity):
                for col_idx, col in enumerate(row):
                    prev_depth[row_idx, col_idx] = (focal_length * baseline) / float(prev_disparity[row_idx, col_idx])
            for row_idx, row in enumerate(next_disparity):
                for col_idx, col in enumerate(row):
                    next_depth[row_idx, col_idx] = (focal_length * baseline) / float(next_disparity[row_idx, col_idx])

            original_disparity = prev_disparity
            for rowi in range(0, len(prev_disparity[:, 1])):
                for coli in range(0, len(prev_disparity[1, :])):
                    prev_disparity[rowi, coli] = int(float(prev_disparity[rowi, coli]))
            prev_disparity = prev_disparity.astype('uint8')
            if print_disparity == 1:
                plt.title('disparity output')
                plt.imshow(prev_disparity)
                plt.show()
            prev_disparity = original_disparity
            # calculate prev and next xyz from xy
            for i in range(num_feats):
                prev_foo = kp1[xy_matches[i].queryIdx].pt
                next_foo = kp2[xy_matches[i].trainIdx].pt
                prev_xy[i, 0] = prev_foo[0]
                prev_xy[i, 1] = prev_foo[1]
                next_xy[i, 0] = next_foo[0]
                next_xy[i, 1] = next_foo[1]
                #old method

                norm_prev_xy = np.array([(prev_xy[i, 0]), (prev_xy[i, 1]), 1])
                norm_next_xy = np.array([(next_xy[i, 0]), (next_xy[i, 1]), 1])
                # norm_prev_xy = np.array([(prev_xy[i, 0]), 480 - (prev_xy[i, 1]), 1])
                # norm_next_xy = np.array([(next_xy[i, 0]), 480 - (next_xy[i, 1]), 1])
                norm_prev_xy = np.reshape(norm_prev_xy, [1, 3])
                norm_next_xy = np.reshape(norm_next_xy, [1, 3])
                prev_foo = np.linalg.inv(K) @ np.transpose(norm_prev_xy)
                next_foo = np.linalg.inv(K) @ np.transpose(norm_next_xy)


                Z = prev_depth[round(prev_xy[i, 1]), round(prev_xy[i, 0])]
                prev_xyz[i, :] = [float(prev_foo[0]) * Z, float(prev_foo[1]) * Z, Z]
                Z = next_depth[round(next_xy[i, 1]), round(next_xy[i, 0])]
                next_xyz[i, :] = [float(next_foo[0]) * Z, float(next_foo[1]) * Z, Z]


            max_inlier_percentage = 0.0
            best_T = np.zeros([4, 4])
            ransac_threshold = 100
            prev_xyz = np.c_[prev_xyz, np.ones(prev_xyz.shape[0])]
            next_xyz = np.c_[next_xyz, np.ones(next_xyz.shape[0])]
            prev_xyz = np.c_[prev_xyz, np.arange(start=0, stop=prev_xyz.shape[0], step=1)]
            next_xyz = np.c_[next_xyz, np.arange(start=0, stop=prev_xyz.shape[0], step=1)]


            full_prev_xyz = prev_xyz
            full_next_xyz = next_xyz

            chosen_index = []

            # Ransac to calculate most accurate T
            if random_ransac != 1:  # If were gonna try every combination
                iter_count = list(itertools.combinations(list(range(0, num_feats)), 2))
                ransac_iterations = len(iter_count)
                bb_text = '# of ransac iterations to try every combination: ' + str(ransac_iterations)
                print(bb_text, end="\r")

            for ransac_idx in range(ransac_iterations):
            # for ransac_idx in range(1):
                inlier_count = 0
                outlier_count = 0
                if ransac_idx != 0 and random_ransac == 1: # if not first iteration, randomly sample n points from the array
                    rand_nums = np.random.randint(num_feats, size=2)
                    prev_xyz = full_prev_xyz[rand_nums, 0:3]
                    next_xyz = full_next_xyz[rand_nums, 0:3]
                elif ransac_idx != 0 and random_ransac != 1:
                    prev_xyz = full_prev_xyz[iter_count[ransac_idx], 0:3]
                    next_xyz = full_next_xyz[iter_count[ransac_idx], 0:3]


                # A = np.transpose(prev_xyz[:, 0:3])
                A = np.transpose(prev_xyz[:, 0:3])
                B = np.transpose(next_xyz[:, 0:3])
                centroid_A = np.mean(A, axis=1)
                centroid_B = np.mean(B, axis=1)

                # ensure centroids are 3x1
                centroid_A = centroid_A.reshape(-1, 1)
                centroid_B = centroid_B.reshape(-1, 1)

                # subtract mean
                Am = A - centroid_A
                Bm = B - centroid_B
                H = Am @ np.transpose(Bm)
                U, S, Vt = np.linalg.svd(H)
                R = Vt.T @ U.T

                # special reflection case
                if np.linalg.det(R) < 0:
                    # print("det(R) < R, reflection detected!, correcting for it ...")
                    Vt[2, :] *= -1
                    R = Vt.T @ U.T

                t = -R @ centroid_A + centroid_B

                T = np.zeros([4, 4])
                T[0:3, 0:3] = R[0:3, 0:3]
                T[0:3, 3:] = t[0:3]
                T[3, 3] = 1

                # calculate average error
                errors = 0

                if ransac_idx == 0:
                    for i in range(full_prev_xyz.shape[0]):
                        full_proj_xyz = T @ full_prev_xyz[i, 0:4].T
                        error_foo = full_next_xyz[i, 0:4] - full_proj_xyz
                        errors += np.sqrt(np.sum(np.square(error_foo)))
                    ransac_threshold = errors/(1.5*full_prev_xyz.shape[0])
                    # print('initial average error: ', ransac_threshold * 1.5)
                    if manual_threshold > 0:
                        ransac_threshold = manual_threshold
                    # print("threshold chosen as: ", ransac_threshold)
                    # print("initial T: \n", T)
                else:
                    foo_chosen_prev = []
                    for i in range(num_feats):
                        full_proj_xyz = T @ full_prev_xyz[i, 0:4].T
                        error_foo = full_next_xyz[i, 0:4] - full_proj_xyz
                        error = np.sqrt(np.sum(np.square(error_foo)))
                        # if error > 1:
                            # print('high error: ', error)
                        if error < ransac_threshold:
                            inlier_count += 1
                            foo_chosen_prev.append(int(full_next_xyz[i, 4]))
                        else:
                            outlier_count += 1
                    if inlier_count/float(inlier_count + outlier_count) > max_inlier_percentage:
                        best_T = T
                        max_inlier_percentage = inlier_count/float(inlier_count + outlier_count)
                        chosen_index = foo_chosen_prev
            # best_T = T
            # print("Final T: \n", best_T)
            # print('Max inliner percentage: ', max_inlier_percentage)
            inlier_values.append(max_inlier_percentage)

            if print_match_figures == 1:
                filtered_matches = []
                for chosen_i in chosen_index:
                    filtered_matches.append(xy_matches[chosen_i])

                foo_img = cv.drawMatches(img1 = image1,
                                            keypoints1 = kp1,
                                            img2 = image2,
                                            keypoints2 = kp2,
                                            matches1to2 = filtered_matches,
                                            outImg = None,
                                            flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                save_figures.saveMatcher(output = foo_img,
                                         matcher = matcher,
                                         descriptor = descriptor,
                                         ransac = "yes")

            # print('best T: \n', T)
            best_T[1, :] = [0, 1, 0, 0]
            error=0
            for a in range(3):
                for aa in range(4):
                    if best_T[a, aa] > 3.0:
                        print('Transformation too large: \n', best_T)
                        print(best_T[a, aa])
                        error=best_T[a, aa]
                        break
                    output_arr.append(best_T[a, aa])
            if error!=0:
                break

        inlier_output_name = output_name = 'iterated_results/withoutmask_feature type_'+descriptor+'_feature matcher_'+matcher+'_num feats_'+str(num_feats)+'_threshold_'+str(manual_threshold)
        inlier_output_name = inlier_output_name+'_inlier values.txt'
        output_name = output_name +'_T.txt'

        inlier_values = np.array(inlier_values)
        output_stats = np.array([np.mean(inlier_values), np.std(inlier_values)])
        output_stats.tofile(inlier_output_name, sep=' ', format='%2.4f')

        output_arr = np.array(output_arr)
        output_arr.tofile(output_name, sep=' ', format='%2.6f')
print("finished!")

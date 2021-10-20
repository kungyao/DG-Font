""" A plug and play Spatial Transformer Module in Pytorch """ 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class SpatialTransformer(nn.Module):
    """
    Implements a spatial transformer 
    as proposed in the Jaderberg paper. 
    Comprises of 3 parts:
    1. Localization Net
    2. A grid generator 
    3. A roi pooled module.
    The current implementation uses a very small convolutional net with 
    2 convolutional layers and 2 fully connected layers. Backends 
    can be swapped in favor of VGG, ResNets etc. TTMV
    Returns:
    A roi feature map with the same input spatial dimension as the input feature map. 
    """
    def __init__(self, in_channels, image_size, fill_background=False, use_dropout=False):
        super(SpatialTransformer, self).__init__()
        self._h, self._w = image_size, image_size
        self._in_ch = in_channels 
        self.dropout = use_dropout
        self.fill_background = fill_background

        # localization net 
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=True) # size : [1x3x32x32]
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.fc1 = nn.Linear(32*(self._h//8)*(self._w//8), 1024)
        self.fc2 = nn.Linear(1024, 6)
        
        # Initialize the weights/bias with identity transformation
        self.fc2.weight.data.zero_()
        self.fc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x): 
        """
        Forward pass of the STN module. 
        x -> input feature map 
        """
        theta = F.relu(self.conv1(x))
        theta = F.relu(self.conv2(theta))
        theta = F.max_pool2d(theta, 2)
        theta = F.relu(self.conv3(theta))
        theta = F.max_pool2d(theta, 2)
        theta = F.relu(self.conv4(theta))
        theta = F.max_pool2d(theta, 2)
        # print("Pre view size:{}".format(x.size()))
        theta = theta.view(x.size(0), -1)
        if self.dropout:
            theta = F.dropout(self.fc1(theta), p=0.5)
            theta = F.dropout(self.fc2(theta), p=0.5)
        else:
            theta = self.fc1(theta)
            theta = self.fc2(theta) # params [Nx6]
        
        theta = theta.view(-1, 2, 3) # change it to the 2x3 matrix 
        # Block offset
        theta[:, :, 2] = 0
        # print(theta)
        # print(x.size())
        affine_grid_points = F.affine_grid(theta, x.size(), align_corners=False)
        assert(affine_grid_points.size(0) == x.size(0)), "The batch sizes of the input images must be same as the generated grid."
        rois = F.grid_sample(x, affine_grid_points, padding_mode="zeros")
        if self.fill_background:
            # F.grid_sample padding mode do not have white option.
            # Input background is white, so we fill rois with white after grid_sample.
            background = F.grid_sample(torch.ones(x.size(), device=x.device), affine_grid_points)
            rois = rois + (1 - background) * torch.max(x)
        # print("rois found to be of size:{}".format(rois.size()))
        # return rois, affine_grid_points, theta
        return rois

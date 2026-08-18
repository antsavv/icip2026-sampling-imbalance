import torch
import torch.nn as nn
import torch.nn.functional as F
import helper_torch_utils as pt_utils
from helper_tool import DataProcessing as DP
import numpy as np
from sklearn.metrics import confusion_matrix

class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        if(config.name == 'S3DIS'):
            weighting_scheme = getattr(config, 'weights', 'none')
            self.class_weights = DP.get_class_weights('S3DIS', weighting_scheme)
            self.fc0 = nn.Linear(6, 8)
            self.fc0_acti = nn.LeakyReLU()
            # Note: TensorFlow momentum=0.99 corresponds to PyTorch momentum=0.01
            self.fc0_bath = nn.BatchNorm1d(8, eps=1e-6, momentum=0.01)
            nn.init.constant_(self.fc0_bath.weight, 1.0)
            nn.init.constant_(self.fc0_bath.bias, 0)
            
            
        elif(config.name == 'DALES'):
            weighting_scheme = getattr(config, 'weights', 'none')
            self.class_weights = DP.get_class_weights('DALES', weighting_scheme)
            self.fc0 = nn.Linear(3, 8)  # 3 input features (XYZ only, no color)
            self.fc0_acti = nn.LeakyReLU()
            # Note: TensorFlow momentum=0.99 corresponds to PyTorch momentum=0.01
            self.fc0_bath = nn.BatchNorm1d(8, eps=1e-6, momentum=0.01)
            nn.init.constant_(self.fc0_bath.weight, 1.0)
            nn.init.constant_(self.fc0_bath.bias, 0)

        elif(config.name == 'STPLS3D'):
            weighting_scheme = getattr(config, 'weights', 'none')
            self.class_weights = DP.get_class_weights('STPLS3D', weighting_scheme)
            self.fc0 = nn.Linear(6, 8)  # 6 input features (XYZ + RGB)
            self.fc0_acti = nn.LeakyReLU()
            # Note: TensorFlow momentum=0.99 corresponds to PyTorch momentum=0.01
            self.fc0_bath = nn.BatchNorm1d(8, eps=1e-6, momentum=0.01)
            nn.init.constant_(self.fc0_bath.weight, 1.0)
            nn.init.constant_(self.fc0_bath.bias, 0)

        self.dilated_res_blocks = nn.ModuleList()       # LFA encoder part
        d_in = 8
        for i in range(self.config.num_layers):
            d_out = self.config.d_out[i]
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out                      # Multiply by 2 because each LFA output is 2*d_out (actual output feature dimension is 2*d_out)

        d_out = d_in
        self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)       # MLP with input 1024 and output 1024 (the middle layer MLP)

        self.decoder_blocks = nn.ModuleList()       # Upsampling decoder part
        for j in range(self.config.num_layers):
            # if j < 4:                                       
            #     d_in = d_out + 2 * self.config.d_out[-j-2]          # -2 because last layer doesn't need concatenation, multiply by 2 because actual output dim is 2*d_out # d_in=1024+512, dim increases due to concat
            #     d_out = 2 * self.config.d_out[-j-2]                 # Adjust to corresponding layer dimension through decoder MLP
            # else:
            #     d_in = 4 * self.config.d_out[-5]            # First d_out used twice, 4*16=64 because 64=32+32, concatenation of two 32s
            #     d_out = 2 * self.config.d_out[-5]           # Adjust output dimension to 32
            # self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True))
            
            if j < config.num_layers - 1:                                       
                d_in = d_out + 2 * self.config.d_out[-j-2]          # -2 because last layer dimension doesn't need concatenation, multiply by 2 because actual output dimension is 2*d_out # d_in=1024+512, dimension increases due to concatenation
                d_out = 2 * self.config.d_out[-j-2]                 # Adjust to corresponding layer dimension through decoder MLP
            else:
                d_in = 4 * self.config.d_out[-config.num_layers]            # First d_out is used twice, 4*16=64 because 64=32+32, concatenation of two 32s
                d_out = 2 * self.config.d_out[-config.num_layers]           # Adjust output dimension to 32
            self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True))
            

        self.fc1 = pt_utils.Conv2d(d_out, 64, kernel_size=(1,1), bn=True)
        self.fc2 = pt_utils.Conv2d(64, 32, kernel_size=(1,1), bn=True)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = pt_utils.Conv2d(32, self.config.num_classes, kernel_size=(1,1), bn=False, activation=None)

    def forward(self, end_points):

        features = end_points['features']  # Batch*channel*npoints
        features = self.fc0(features)

        # The following three lines were modified later
        features = self.fc0_acti(features)
        features = features.transpose(1,2)
        features = self.fc0_bath(features)

        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1 # Add a dimension to use 2D [1,1] convolution

        # ###########################Encoder############################
        f_encoder_list = []         # Store features after each LFA for later concatenation
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](features, end_points['xyz'][i], end_points['neigh_idx'][i])    # Need to use neighbor indices

            f_sampled_i = self.random_sample(f_encoder_i, end_points['sub_idx'][i])
            features = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)      # First time add features before downsampling, feature dim is 32, used twice in decoder
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        features = self.decoder_0(f_encoder_list[-1])   # Middle layer MLP

        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(features, end_points['interp_idx'][-j - 1])                 # First perform interpolation
            f_decoder_i = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))        # Concatenate with previous features

            features = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################

        features = self.fc1(features)
        features = self.fc2(features)
        features = self.dropout(features)
        features = self.fc3(features)
        f_out = features.squeeze(3)

        end_points['logits'] = f_out
        return end_points

    @staticmethod
    def random_sample(feature, pool_idx):       # Since indices are already saved, random sampling just reads the index values
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)    # batch*channel*npoints   # Remove one dimension
        num_neigh = pool_idx.shape[-1]      # Number of KNN neighbors
        d = feature.shape[1]                # Feature dimension
        batch_size = pool_idx.shape[0]      # pool_idx dimension is [6, 10240, 16], where 16 is the index of 16 neighbors
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))  # Get features of sampled points
        # First expand pool_idx with a middle feature dimension, after expansion: [batch, 1, npoints*nsamples]
        # Then repeat each row in each batch feature.shape[1]-1 times (to reach feature.shape[1] dim), after repeat: [batch, feature.shape[1], npoints*nsamples]
        # Then index the feature tensor according to the processed pool_idx
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1  [0] gets values, [1] gets indices; max means taking the maximum feature among 16 nearest neighbors for each feature dimension
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1,feature.shape[1],1))  # Find features of points to upsample to
        # (Key point is the ordered nature of data matrix, allowing features to be propagated back to points before previous sampling)
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features



def compute_acc(end_points):

    logits = end_points['valid_logits']
    labels = end_points['valid_labels']
    logits = logits.max(dim=1)[1]
    acc = (logits == labels).sum().float() / float(labels.shape[0])
    end_points['acc'] = acc
    return acc, end_points


class IoUCalculator:
    def __init__(self, cfg):
        self.gt_classes = [0 for _ in range(cfg.num_classes)]               # Initialize a list of length num_classes with all zeros
        self.positive_classes = [0 for _ in range(cfg.num_classes)]         # Same as above  
        self.true_positive_classes = [0 for _ in range(cfg.num_classes)]
        self.cfg = cfg

    def add_data(self, end_points):
        logits = end_points['valid_logits']     # Logits after ignoring labels        # Dimension is (40960*batch_size)
        labels = end_points['valid_labels']     # Labels after ignoring labels
        pred = logits.max(dim=1)[1]             # [1] selects the second element of max object; this object has length 2: first is max value, second is max index
        pred_valid = pred.detach().cpu().numpy()
        labels_valid = labels.detach().cpu().numpy()

        val_total_correct = 0       # This variable seems unused?
        val_total_seen = 0

        correct = np.sum(pred_valid == labels_valid)    # Count correctly classified points
        val_total_correct += correct    # Accumulate correct count
        val_total_seen += len(labels_valid) # Accumulate total count

        # Compute confusion matrix (columns are predicted classes, rows are true classes, describes correct and misclassified counts)
        conf_matrix = confusion_matrix(labels_valid, pred_valid, labels=np.arange(0, self.cfg.num_classes, 1)) 
        self.gt_classes += np.sum(conf_matrix, axis=1)      # Sum by rows, represents total ground truth data points for each class
        self.positive_classes += np.sum(conf_matrix, axis=0)    # Sum by columns, represents total predicted data points for each class
        self.true_positive_classes += np.diagonal(conf_matrix)  # Extract diagonal elements

    def compute_iou(self):
        iou_list = []
        for n in range(0, self.cfg.num_classes, 1):
            if float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n]) != 0:       # This is the denominator, ensure it's not zero
                iou = self.true_positive_classes[n] / float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n])  # Compute IoU for class n
                iou_list.append(iou)
            else:
                iou_list.append(0.0)            # Denominator is zero only when all three are zero, so IoU=0
        mean_iou = sum(iou_list) / float(self.cfg.num_classes)  # Divide by number of classes
        return mean_iou, iou_list



class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.mlp1 = pt_utils.Conv2d(d_in, d_out//2, kernel_size=(1,1), bn=True)
        self.lfa = Building_block(d_out)
        self.mlp2 = pt_utils.Conv2d(d_out, d_out*2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in, d_out*2, kernel_size=(1,1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)  # Batch*channel*npoints*1                # The blue MLP in the diagram
        f_pc = self.lfa(xyz, f_pc, neigh_idx)  # Batch*d_out*npoints*1      # This LFA contains two local spatial encodings and two attention poolings
        f_pc = self.mlp2(f_pc)                                              # The blue MLP after
        shortcut = self.shortcut(feature)                                   # The MLP below
        return F.leaky_relu(f_pc+shortcut, negative_slope=0.2)              # Element-wise addition


class Building_block(nn.Module):
    def __init__(self, d_out):  #  d_in = d_out//2
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(10, d_out//2, kernel_size=(1,1), bn=True)
        self.att_pooling_1 = Att_pooling(d_out, d_out//2)

        self.mlp2 = pt_utils.Conv2d(d_out//2, d_out//2, kernel_size=(1, 1), bn=True)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):  # feature: Batch*channel*npoints*1
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  # batch*npoint*nsamples*10  # The 10 features here are fixed
        f_xyz = f_xyz.permute((0, 3, 1, 2))  # batch*10*npoint*nsamples  # Swap tensor dimensions
        f_xyz = self.mlp1(f_xyz)            # Encode spatial features, corresponds to position encoding in diagram
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel Get features of K nearest neighbors
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples Adjust dimensions
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)      # Concatenate feature information and spatial information
        f_pc_agg = self.att_pooling_1(f_concat)  # Batch*channel*npoints*1

        f_xyz = self.mlp2(f_xyz)        # Encode spatial information again using previously encoded spatial info
        f_neighbours = self.gather_neighbour(f_pc_agg.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples Adjust dimensions
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # batch*npoint*nsamples*3

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)  # batch*npoint*nsamples*3  This step is like broadcasting, enabling direct subtraction in next line; result is center point's own xyz matrix corresponding to pi in the paper
        relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3   # Subtract neighbor coords from own coords to compute relative coordinates
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))  # batch*npoint*nsamples*1   # Relative distance to center point
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)  # batch*npoint*nsamples*10
        return relative_feature

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel(xyz or features)
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)      # This gather is hard to understand, think carefully
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))     # From original point xyz coords (or features), find coords (or features) of 16 nearest neighbors (note: pc matrix is ordered, its indices relate to neighbor_idx)
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel     # This is the 16 nearest neighbor coords for each of the 40960 points
        return features


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)     # Attention pooling also has an MLP to change output shape

    def forward(self, feature_set):

        att_activation = self.fc(feature_set)           # Pass concatenated matrix through FC + softmax to learn attention scores of same dimension
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores                # Element-wise multiplication
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)   # Sum
        f_agg = self.mlp(f_agg)                         # Final MLP to adjust dimensions
        return f_agg


def compute_loss(end_points, cfg, device):

    logits = end_points['logits']       # Get logits and labels from network
    labels = end_points['labels']

    logits = logits.transpose(1, 2).reshape(-1, cfg.num_classes)        # Flatten batch dimension, move data to point dimension
    labels = labels.reshape(-1)

    # Boolean mask of points that should be ignored
    # ignored_bool = labels == 0                              # Label 0 was masked here
    # for ign_label in cfg.ignored_label_inds:
    #     ignored_bool = ignored_bool | (labels == ign_label)

    # ignored_bool = labels == 0                              
    ignored_bool = torch.zeros(len(labels), dtype=torch.bool).to(device)
    for ign_label in cfg.ignored_label_inds:                                    # No problem here, issue is below
        ignored_bool = ignored_bool | (labels == ign_label)

    # Collect logits and labels that are not ignored
    valid_idx = ignored_bool == 0
    valid_logits = logits[valid_idx, :]
    valid_labels_init = labels[valid_idx]

    # Reduce label values in the range of logit shape
    reducing_list = torch.arange(0, cfg.num_classes).long().to(device)       
    inserted_value = torch.zeros((1,)).long().to(device)
    for ign_label in cfg.ignored_label_inds:
        reducing_list = torch.cat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
    valid_labels = torch.gather(reducing_list, 0, valid_labels_init)            # This operation is hard to understand
    
    # Use custom loss function if provided, otherwise use default weighted CE
    loss_fn = getattr(cfg, 'loss_fn', None)
    if loss_fn is not None:
        loss = loss_fn(valid_logits, valid_labels)
    else:
        loss = get_loss(valid_logits, valid_labels, cfg.class_weights, device)
    
    end_points['valid_logits'], end_points['valid_labels'] = valid_logits, valid_labels     # valid_logits is the logits after ignoring labels
    end_points['loss'] = loss
    return loss, end_points


def get_loss(logits, labels, pre_cal_weights, device):
    # calculate the weighted cross entropy according to the inverse frequency
    class_weights = torch.from_numpy(pre_cal_weights).float().to(device).reshape(-1)
    
    if len(class_weights) == 0:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    output_loss = criterion(logits, labels)
    return output_loss
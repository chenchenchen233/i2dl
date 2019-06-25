from torch.utils.data import Dataset, DataLoader
import matplotlib.image as mpimg
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            custom_point (list): which points to train on
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.key_pts_frame.dropna(inplace=True)
        self.key_pts_frame.reset_index(drop=True, inplace=True)
# =============================================================================
#         key_pts_frame=self.key_pts_frame
#         image=self.key_pts_frame.loc[2,'Image']
# =============================================================================
        
        self.transform = transform
        

    def __len__(self):
        ########################################################################
        # TODO:                                                                #
        # Return the length of the dataset                                     #
        ########################################################################
        return len(self.key_pts_frame)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
   #from exercise_code.data_utils import get_keypoints
   #from exercise_code.data_utils import get_image
    
    def __getitem__(self, idx):
        ########################################################################
        # TODO:                                                                #
        # Return the idx sample in the Dataset. A sample should be a dictionary#
        # where the key, value should be like                                  #
        #        {'image': image of shape [C, H, W],                           #
        #         'keypoints': keypoints of shape [num_keypoints, 2]}          #
        # You can use mpimg.imread(image path) to read out image data          #
        ########################################################################
        image_string = self.key_pts_frame.loc[idx]['Image']
        #a=np.array([int(item) for item in image_string.split()]).reshape((96, 96))
        sample={}

        sample['image']=np.array([int(item) for item in image_string.split()]).reshape((1, 96, 96))
        
        #sample['image']=string2image(image_string)
        #sample['image']=get_keypoints(idx, self.key_pts_frame)
        keypoint_cols = list(self.key_pts_frame.columns)[:-1]
        sample['keypoints']=np.array(self.key_pts_frame.iloc[idx][keypoint_cols].values.reshape((15, 2)),dtype=np.float)
        #sample['keypoints']=get_image(idx, self.key_pts_frame)
        
# =============================================================================
#                 ###test show image with keypoints
#         image=sample['image']
#         
#         predicted_key_pts=sample['keypoints']
#         plt.imshow(image, cmap='gray')
#         
#         plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=80, marker='.', c='m')
#         plt.show()
#         ##end
# =============================================================================
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        if self.transform:
            sample = self.transform(sample)

        return sample
    
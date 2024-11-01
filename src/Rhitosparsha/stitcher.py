import glob
import cv2
import os
import numpy as np

class PanaromaStitcher():
    def __init__(self):
        pass

    def detect_and_match_keypoints(self, img1, img2):
        """Detects keypoints using SIFT and matches them using BFMatcher."""
        sift = cv2.SIFT_create()
        
        # Detect SIFT keypoints and descriptors
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

        # Use BFMatcher to match descriptors
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract location of matched keypoints
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        return points1, points2

    def compute_homography(self, pts1, pts2):
        """Computes the homography matrix from matched points using DLT."""
        assert pts1.shape == pts2.shape and pts1.shape[0] >= 4, "Need at least 4 point correspondences"
        
        num_points = pts1.shape[0]
        A = []

        for i in range(num_points):
            x1, y1 = pts1[i][0], pts1[i][1]
            x2, y2 = pts2[i][0], pts2[i][1]
            A.append([-x1, -y1, -1, 0, 0, 0, x2*x1, x2*y1, x2])
            A.append([0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2])

        A = np.array(A)
        U, S, Vt = np.linalg.svd(A)
        H = Vt[-1, :].reshape(3, 3)
        
        # Normalize the homography matrix
        H /= H[2, 2]
        return H

    def warp_images(self, img1, img2, H):
        """Warp img2 to img1 using the homography matrix H."""
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]

        # Get the corners of img2
        corners2 = np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners2, H)

        # Combine the corners from both images to get the size of the final panorama
        corners = np.concatenate((np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2), 
                                  warped_corners), axis=0)

        # Calculate the size of the output panorama
        [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)

        # Adjust the translation homography
        translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])  # translation matrix

        # Warp the second image and place it in the panorama
        result = cv2.warpPerspective(img2, translation @ H, (xmax - xmin, ymax - ymin))
        result[-ymin:height1 - ymin, -xmin:width1 - xmin] = img1

        return result

    def make_panaroma_for_images_in(self, path):
        """Main function to stitch images and return the final panorama and homography matrices."""
        all_images = sorted(glob.glob(path + os.sep + '*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        # Read the first image
        img1 = cv2.imread(all_images[0])
        stitched_image = img1
        homography_matrix_list = []

        # Loop through the rest of the images and stitch them together
        for i in range(1, len(all_images)):
            img2 = cv2.imread(all_images[i])

            # Step 1: Detect and match keypoints
            points1, points2 = self.detect_and_match_keypoints(stitched_image, img2)

            # Step 2: Compute homography from matched keypoints
            H = self.compute_homography(points1, points2)
            homography_matrix_list.append(H)

            # Step 3: Warp the images and stitch them
            stitched_image = self.warp_images(stitched_image, img2, H)

        # Return final panorama and homography matrices
        return stitched_image, homography_matrix_list

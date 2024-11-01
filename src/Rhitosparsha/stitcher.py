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

    def bilinear_interpolation(self, img, x, y):
        """Performs bilinear interpolation for non-integer pixel coordinates."""
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = x0 + 1, y0 + 1

        # Ensure the coordinates are within the image bounds
        if x0 >= img.shape[1] - 1 or y0 >= img.shape[0] - 1 or x0 < 0 or y0 < 0:
            return [0, 0, 0]

        # Get pixel values at the corners
        I00 = img[y0, x0]
        I10 = img[y0, x1]
        I01 = img[y1, x0]
        I11 = img[y1, x1]

        # Compute the interpolation weights
        dx = x - x0
        dy = y - y0

        # Perform bilinear interpolation
        interpolated = (1 - dx) * (1 - dy) * I00 + dx * (1 - dy) * I10 + (1 - dx) * dy * I01 + dx * dy * I11
        return interpolated.astype(np.uint8)
    
    def warp_images(self, img1, img2, H):
        """Warp img2 to img1 using the homography matrix H (without cv2.warpPerspective)."""
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
        H_inv = np.linalg.inv(translation @ H)  # Inverse of the homography

        # Create an empty image for the panorama
        panorama_height, panorama_width = ymax - ymin, xmax - xmin
        panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)

        # Copy img1 to the panorama at the correct offset
        panorama[-ymin:height1 - ymin, -xmin:width1 - xmin] = img1

        # Warp img2 to the panorama using the inverse homography matrix
        for y in range(panorama_height):
            for x in range(panorama_width):
                # Transform the panorama coordinates (x, y) back to img2's coordinates
                warped_point = np.dot(H_inv, np.array([x, y, 1]))
                warped_point /= warped_point[2]  # Normalize by the homogeneous coordinate
                x2, y2 = warped_point[0], warped_point[1]

                # Check if the coordinates are within the bounds of img2
                if 0 <= x2 < width2 and 0 <= y2 < height2:
                    # Perform bilinear interpolation
                    panorama[y, x] = bilinear_interpolation(img2, x2, y2)

        return panorama

    
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

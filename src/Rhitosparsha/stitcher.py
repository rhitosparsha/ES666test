import pdb
import glob
import cv2
import os
import numpy as np

class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self, path):
        imf = path
        all_images = sorted(glob.glob(imf + os.sep + '*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        # Step 1: Load images
        images = [cv2.imread(img) for img in all_images]

        # Step 2: Detect keypoints and descriptors using SIFT
        sift = cv2.SIFT_create()
        keypoints_descriptors = [sift.detectAndCompute(image, None) for image in images]

        # Step 3: Match keypoints between consecutive images using BFMatcher
        bf = cv2.BFMatcher()
        matches = [bf.knnMatch(keypoints_descriptors[i][1], keypoints_descriptors[i+1][1], k=2)
                   for i in range(len(images) - 1)]

        # Step 4: Filter good matches using Lowe's ratio test
        good_matches = []
        for match in matches:
            good = []
            for m, n in match:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            good_matches.append(good)

        # Step 5: Estimate homographies from scratch
        homography_matrix_list = []
        for i, good_match in enumerate(good_matches):
            src_pts = np.float32([keypoints_descriptors[i][0][m.queryIdx].pt for m in good_match]).reshape(-1, 2)
            dst_pts = np.float32([keypoints_descriptors[i+1][0][m.trainIdx].pt for m in good_match]).reshape(-1, 2)

            H = self.estimate_homography(src_pts, dst_pts)
            homography_matrix_list.append(H)

        # Step 6: Warp images and stitch them together
        stitched_image = images[0]
        for i in range(1, len(images)):
            H = homography_matrix_list[i-1]
            stitched_image = self.manual_warp_and_stitch(stitched_image, images[i], H)

        return stitched_image, homography_matrix_list

    def estimate_homography(self, src_pts, dst_pts):
        # Use DLT to estimate homography matrix
        A = []
        for i in range(len(src_pts)):
            x, y = src_pts[i][0], src_pts[i][1]
            u, v = dst_pts[i][0], dst_pts[i][1]
            A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
            A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])

        A = np.array(A)
        _, _, V = np.linalg.svd(A)
        H = V[-1].reshape(3, 3)

        # Normalize H
        H = H / H[2, 2]
        return H

    def manual_warp_and_stitch(self, image1, image2, H):
        # Warp image2 manually using the homography matrix H
        height1, width1 = image1.shape[:2]
        height2, width2 = image2.shape[:2]

        # Get corners of image2
        corners_image2 = np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)

        # Warp corners of image2 to the plane of image1
        warped_corners = self.apply_homography(H, corners_image2)

        # Find the size of the panorama by combining corners of both images
        all_corners = np.vstack((np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]), 
                                 warped_corners.reshape(-1, 2)))

        [x_min, y_min] = np.int32(all_corners.min(axis=0) - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0) + 0.5)

        translation = [-x_min, -y_min]

        # Create an empty canvas for the panorama
        panorama = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)

        # Paste image1 in the panorama
        panorama[translation[1]:height1 + translation[1], translation[0]:width1 + translation[0]] = image1

        # Now, manually warp image2 and blend it into the panorama
        H_inv = np.linalg.inv(H)

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                # Apply inverse homography to find the corresponding source pixel in image2
                src_coords = np.array([x, y, 1]).reshape(3, 1)
                dst_coords = np.dot(H_inv, src_coords)
                dst_coords = dst_coords / dst_coords[2]  # Normalize to get homogeneous coordinates

                x_src, y_src = int(dst_coords[0]), int(dst_coords[1])

                if 0 <= x_src < width2 and 0 <= y_src < height2:
                    # If the coordinates are within the bounds of image2, copy the pixel
                    panorama[y - y_min, x - x_min] = image2[y_src, x_src]

        return panorama

    def apply_homography(self, H, points):
        # Apply homography matrix H to a set of points
        points = np.concatenate([points, np.ones((points.shape[0], 1, 1))], axis=2)
        transformed_points = np.dot(H, points.transpose(0, 2, 1)).transpose(0, 2, 1)
        transformed_points /= transformed_points[:, :, 2].reshape(-1, 1, 1)
        return transformed_points[:, :, :2]

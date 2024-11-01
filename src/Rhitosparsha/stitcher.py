import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

class PanaromaStitcher:
    def __init__(self, image_files, focal_length=800, apply_cylindrical_warp=False):
        self.focal_length = focal_length
        self.images = [self.cylindrical_warp(cv2.imread(img), focal_length) for img in image_files] if apply_cylindrical_warp else [cv2.imread(img) for img in image_files]
        self.keypoints = []
        self.descriptors = []
        self.translation_left_matrices = []
        self.translation_right_matrices = []

        # Initialize SIFT detector
        self.sift = cv2.SIFT_create()
        
        # Detect keypoints and descriptors for all images
        for img in self.images:
            kp, des = self.sift.detectAndCompute(img, None)
            self.keypoints.append(kp)
            self.descriptors.append(des)

    def cylindrical_warp(self, image, focal_length):
        h, w = image.shape[:2]
        x_c, y_c = w // 2, h // 2  # Center coordinates

        # Create a mesh grid for image coordinates
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Convert to cylindrical coordinates
        theta = (u - x_c) / focal_length
        h_cyl = (v - y_c) / focal_length

        x_hat = np.sin(theta)
        y_hat = h_cyl
        z_hat = np.cos(theta)

        # Project cylindrical coordinates back to image plane
        x_img = (focal_length * x_hat / z_hat + x_c).astype(np.int32)
        y_img = (focal_length * y_hat / z_hat + y_c).astype(np.int32)

        # Mask to keep points within image bounds
        valid_mask = (x_img >= 0) & (x_img < w) & (y_img >= 0) & (y_img < h)

        # Create the cylindrical image with valid pixel mappings
        cylindrical_img = np.zeros_like(image)
        cylindrical_img[v[valid_mask], u[valid_mask]] = image[y_img[valid_mask], x_img[valid_mask]]

        return cylindrical_img

    def find_matches(self):
        bf = cv2.BFMatcher()
        all_good_matches = []

        # Find matches for consecutive image pairs
        for i in range(len(self.images) - 1):
            matches = bf.knnMatch(self.descriptors[i], self.descriptors[i + 1], k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            all_good_matches.append(good_matches)
        return all_good_matches

    def compute_homography(self, src_pts, dst_pts):
        A = []
        for i in range(len(src_pts)):
            x, y = src_pts[i][0]
            xp, yp = dst_pts[i][0]
            A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
            A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
        A = np.array(A)
        _, _, Vh = np.linalg.svd(A)
        L = Vh[-1, :] / Vh[-1, -1]
        return L.reshape(3, 3)

    def ransac_homography(self, src_pts, dst_pts, num_iterations=1000, threshold=5.0):
        max_inliers = 0
        best_H = None
        inliers_list = []

        for _ in range(num_iterations):
            indices = np.random.choice(len(src_pts), 4, replace=False)
            src_sample = src_pts[indices]
            dst_sample = dst_pts[indices]
            H = self.compute_homography(src_sample, dst_sample)

            src_pts_homogeneous = np.concatenate([src_pts, np.ones((len(src_pts), 1, 1))], axis=2)
            projected_pts = np.dot(H, src_pts_homogeneous.reshape(-1, 3).T).T
            projected_pts = projected_pts[:, :2] / projected_pts[:, 2:]

            distances = np.linalg.norm(projected_pts - dst_pts.reshape(-1, 2), axis=1)
            inliers = distances < threshold
            num_inliers = np.sum(inliers)

            if num_inliers > max_inliers:
                max_inliers = num_inliers
                inliers_list = inliers
        
        src_sample = src_pts[inliers_list]
        dst_sample = dst_pts[inliers_list]
        H = self.compute_homography(src_sample, dst_sample)
        return H

    def all_homographies(self):
        homography_pairs = []
        all_good_matches = self.find_matches()

        for i in range(len(self.images) - 1):
            src_pts = np.float32([self.keypoints[i][m.queryIdx].pt for m in all_good_matches[i]]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.keypoints[i + 1][m.trainIdx].pt for m in all_good_matches[i]]).reshape(-1, 1, 2)
            H = self.ransac_homography(src_pts, dst_pts)
            homography_pairs.append(H)
        
        return homography_pairs

    def make_panorama_images_in(self, homographies, ref_index):
        ref_img = self.images[ref_index]
        panorama = ref_img
        homography_matrix_list = []  # To store all the homography matrices
    
        # Stitch left
        for i in range(ref_index - 1, -1, -1):
            H_inv = np.linalg.inv(homographies[i])
            panorama = self.warp_and_blend(panorama, self.images[i], H_inv)
            homography_matrix_list.append(H_inv)  # Add the inverse homography matrix
    
        # Stitch right
        for i in range(ref_index, len(self.images) - 1):
            panorama = self.warp_and_blend(panorama, self.images[i + 1], homographies[i])
            homography_matrix_list.append(homographies[i])  # Add the homography matrix
    
        return panorama, homography_matrix_list  # Return both panorama and homographies


    def warp_and_blend(self, base_img, img, H):
        h_base, w_base = base_img.shape[:2]
        h_img, w_img = img.shape[:2]

        # Warp img to the base image's perspective
        warped_img = cv2.warpPerspective(img, H, (w_base + w_img, h_base))
        
        # Create a canvas large enough for both images
        panorama = np.zeros((h_base, w_base + w_img, 3), dtype=np.uint8)
        panorama[:h_base, :w_base] = base_img

        # Combine the two images
        panorama[:h_img, :w_img] = np.maximum(panorama[:h_img, :w_img], warped_img[:h_img, :w_img])
        return panorama

    def stitch_images(self):
        center_index = len(self.images) // 2
        all_homographies = self.all_homographies()

        final_panorama = self.make_panorama_images_in(all_homographies, center_index)
        
        # Return both the panorama and the list of homographies
        return final_panorama, all_homographies


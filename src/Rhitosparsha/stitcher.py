import glob
import cv2
import os
import numpy as np

class PanaromaStitcher():
    def __init__(self):
        pass

    def compute_homography(self, src_pts, dst_pts):
        # Compute homography matrix using DLT method given source and destination points.
        num_points = src_pts.shape[0]
        A = []
        for i in range(num_points):
            x_src, y_src = src_pts[i][0], src_pts[i][1]
            x_dst, y_dst = dst_pts[i][0], dst_pts[i][1]
            A.append([-x_src, -y_src, -1, 0, 0, 0, x_src * x_dst, y_src * x_dst, x_dst])
            A.append([0, 0, 0, -x_src, -y_src, -1, x_src * y_dst, y_src * y_dst, y_dst])

        A = np.array(A)
        U, S, Vh = np.linalg.svd(A)
        H = Vh[-1].reshape(3, 3)
        return H / H[-1, -1]

    def warp_image(self, img, H, output_shape):
        # Manually warp an image using a given homography matrix H.
        h, w = output_shape[:2]
        warped_image = np.zeros((h, w, 3), dtype=np.uint8)
        H_inv = np.linalg.inv(H)

        for y in range(h):
            for x in range(w):
                p = np.array([x, y, 1])
                mapped_p = H_inv @ p
                mapped_p /= mapped_p[2]                
                x_src, y_src = mapped_p[0], mapped_p[1]
                if 0 <= x_src < img.shape[1] and 0 <= y_src < img.shape[0]:
                    warped_image[y, x] = self.bilinear_interpolate(img, x_src, y_src)

        return warped_image

    def bilinear_interpolate(self, img, x, y):
        # Perform bilinear interpolation to get the pixel value at non-integer (x, y) coordinates.
        x0 = int(np.floor(x))
        x1 = min(x0 + 1, img.shape[1] - 1)
        y0 = int(np.floor(y))
        y1 = min(y0 + 1, img.shape[0] - 1)

        Ia = img[y0, x0]
        Ib = img[y1, x0]
        Ic = img[y0, x1]
        Id = img[y1, x1]

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        return wa * Ia + wb * Ib + wc * Ic + wd * Id
    
    def apply_homography(self, points, H):
        # Manually apply a homography matrix to a set of points
        if points.ndim == 3 and points.shape[1] == 1 and points.shape[2] == 2:
            points = points.reshape(points.shape[0], 2)  # Reshape to (N, 2)

        # Check if points are now 2D with the expected shape
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Points must be of shape (N, 2) but got shape: {}".format(points.shape))

        # Reshape to ensure it's 2D: (N, 2)
        if points.ndim == 2 and points.shape[1] == 2:
            # Good shape, continue
            points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
        else:
            raise ValueError("Points must be of shape (N, 2) or (N, 3) but got shape: {}".format(points.shape))
        transformed_points = (H @ points_homogeneous.T).T
        transformed_points = transformed_points[:, :2] / transformed_points[:, 2, np.newaxis]
        return transformed_points
    
    def stitch_images(self, img1, img2, H):
        # Warp img2 to img1 using the homography H and return the stitched result.
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]
        
        corners_img2 = np.float32([[0, 0], [0, height2-1], [width2-1, height2-1], [width2-1, 0]]).reshape(-1, 1, 2)
        warped_corners = self.apply_homography(corners_img2, H)
        print("Corners of img2:", corners_img2)
        print("Warped corners shape:", warped_corners.shape)  # Debug print
        
        # Calculate the bounding box of the resulting panorama
        img1_corners = np.float32([[0, 0], [0, height1-1], [width1-1, height1-1], [width1-1, 0]])  # Shape (4, 2)
        print("Corners of img1:", img1_corners)
        all_corners = np.vstack((img1_corners, warped_corners))  # Shape (8, 2)
        print("All corners shape:", all_corners.shape)
        
        [x_min, y_min] = np.int32(all_corners.min(axis=0))
        [x_max, y_max] = np.int32(all_corners.max(axis=0))
        
        translation_dist = [-x_min, -y_min]
        H_translate = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])  # Translation matrix
        
        panorama_size = (x_max - x_min, y_max - y_min)
        
        img2_warped = self.warp_image(img2, H_translate @ H, panorama_size)
        
        panorama = np.zeros((panorama_size[1], panorama_size[0], 3), dtype=np.uint8)
        
        panorama[translation_dist[1]:translation_dist[1]+height1, translation_dist[0]:translation_dist[0]+width1] = img1
        
        min_height = min(panorama.shape[0], img2_warped.shape[0])
        min_width = min(panorama.shape[1], img2_warped.shape[1])
        
        panorama[:min_height, :min_width] = np.where(img2_warped[:min_height, :min_width] > 0, img2_warped[:min_height, :min_width], panorama[:min_height, :min_width])
        
        return panorama

    def match_keypoints(self, img1, img2):
        # Detect and match keypoints between two images.
        # Use SIFT to detect keypoints and descriptors
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        # Use BFMatcher to match descriptors
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        return src_pts, dst_pts

    def make_panaroma_for_images_in(self, path):
        imf = path
        all_images = sorted(glob.glob(imf + os.sep + '*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        stitched_image = cv2.imread(all_images[0])
        homography_matrix_list = []

        for i in range(1, len(all_images)):
            img1 = stitched_image
            img2 = cv2.imread(all_images[i])

            src_pts, dst_pts = self.match_keypoints(img1, img2)
            H = self.compute_homography(src_pts, dst_pts)
            homography_matrix_list.append(H)

            stitched_image = self.stitch_images(img1, img2, H)

        return stitched_image, homography_matrix_list
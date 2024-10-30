import glob
import cv2
import os
import numpy as np
import random

class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self, path):
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        img_list = []

        for image in all_images:
            img = cv2.imread(image)
            img_list.append(img)

        homography_matrix_list = []

        mid_image = (len(img_list)-1)//2
        stitched_image = img_list[mid_image]

        transform_left = False

        num_images_stitched = 1
        index_left = mid_image - 1
        index_right = mid_image + 1

        while num_images_stitched != len(img_list):
            if transform_left:
                if index_left < 0:
                    break
                output_img, homography_matrix, transform_left = self.stitch_images(img_list[index_left], stitched_image, transform_left)
                index_left -= 1
            else:
                if index_right >= len(img_list):
                    break
                output_img, homography_matrix, transform_left = self.stitch_images(stitched_image, img_list[index_right], transform_left)
                index_right += 1
            stitched_image = output_img
            homography_matrix_list.append(homography_matrix)
            num_images_stitched += 1

        if num_images_stitched != len(img_list):
            if index_left >= 0:
                for i in range(index_left, -1, -1):
                    output_img, homography_matrix, transform_left = self.stitch_images(img_list[i], stitched_image, transform_left)
                    stitched_image = output_img
                    homography_matrix_list.append(homography_matrix)
                    num_images_stitched += 1
            else:
                for i in range(index_right, len(img_list)):
                    output_img, homography_matrix, transform_left = self.stitch_images(stitched_image, img_list[i], transform_left)
                    stitched_image = output_img
                    homography_matrix_list.append(homography_matrix)
                    num_images_stitched += 1

        stitched_image = self.format_image(stitched_image)
        return stitched_image, homography_matrix_list
    
    def stitch_images(self, left_img, right_img, transform_left):
        # Get the dimensions of both images
        left_shape = left_img.shape
        right_shape = right_img.shape
        
        # Define transformation
        translation_matrix = transform_left
    
        # Apply the transformation to the right image
        transformed_right_img = self.wrap_perspective(right_img, translation_matrix, left_shape)
    
        # Create an output image large enough to hold both images
        output_height = max(left_shape[0], transformed_right_img.shape[0])
        output_width = left_shape[1] + transformed_right_img.shape[1]  # Width combined
    
        output_img = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
        # Position left image in output
        output_img[0:left_shape[0], 0:left_shape[1]] = left_img
    
        # Find the min x and y coordinates of the transformed right image
        y_min, y_max, x_min, x_max = self.find_min_max_coordinates(transformed_right_img)
    
        # Place the transformed right image into the output image
        output_img[-y_min:y_max - y_min, -x_min:x_max - x_min] = transformed_right_img[y_min:y_max, x_min:x_max]
    
        return output_img, translation_matrix


    def get_keypoints(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()

        kp = sift.detect(img, None)
        kp, des = sift.compute(img, kp)
        return kp, des
    
    def get_matched_points(self, kp_left, des_left, kp_right, des_right):
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        pre_matches = bf.knnMatch(des_left, des_right, k=2)

        matches = []
        for m, n in pre_matches:
            if m.distance < 0.75 * n.distance:
                matches.append([kp_left[m.queryIdx].pt, kp_right[m.trainIdx].pt])

        return np.array(matches)
    
    def check_order(self, matches): 
        avg_x_img1 = np.mean(matches[:, 0, 0])
        avg_x_img2 = np.mean(matches[:, 1, 0])

        return avg_x_img1 > avg_x_img2

    
    def ransac(self, matches):
        most_inliers = 0
        best_homography_matrix = None
        threshold = 5
        num_trials = 5000
        sample_size = 6

        for i in range(num_trials):
            random_pts = random.choices(matches, k=sample_size)
            homography_matrix = self.calculate_homography_matrix(random_pts)
            num_inliers = 0
            for match in matches:
                left_image_point = np.array([match[0][0], match[0][1], 1])
                right_image_point = np.array([match[1][0], match[1][1], 1])
                predicted_right_image_point = np.dot(homography_matrix, left_image_point)
                if predicted_right_image_point[2] == 0:
                    continue
                predicted_right_image_point /= predicted_right_image_point[2]
                if np.linalg.norm(right_image_point - predicted_right_image_point) < threshold:
                    num_inliers += 1
            if num_inliers > most_inliers:
                most_inliers = num_inliers
                best_homography_matrix = homography_matrix
        return best_homography_matrix

    def calculate_homography_matrix(self, points):
        A = []
        for point in points:
            x, y = point[0]
            u, v = point[1]
            A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
            A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
        A = np.array(A)
        U, S, V = np.linalg.svd(A)
        H = V[-1].reshape(3, 3)
        H /= H[2, 2]
        return H
    
    def perspective_transform(self, points, homography_matrix):
        transformed_points = []
        for point in points:
            [x, y] = point[0]
            [x_, y_, z_] = np.dot(homography_matrix, [x, y, 1])
            transformed_points.append([[x_/z_, y_/z_]])
        return np.float32(transformed_points)    

    def wrap_perspective(self, img, matrix, output_shape):
        # Get image dimensions
        h, w = img.shape[:2]
    
        # Define the corners of the image
        corners = np.array([[0, 0, 1],
                            [w, 0, 1],
                            [w, h, 1],
                            [0, h, 1]]).T
    
        # Apply the homography transformation
        transformed_coords = matrix @ corners
        transformed_coords /= transformed_coords[2, :]  # Normalize by z-coordinate
    
        # Convert to integer coordinates
        transformed_coords = np.round(transformed_coords).astype(int)
    
        # Create an output image
        output_img = np.zeros(output_shape, dtype=img.dtype)
    
        # Fill the output image with the input image
        valid_mask = (transformed_coords[0, :] >= 0) & (transformed_coords[0, :] < output_shape[1]) & \
                     (transformed_coords[1, :] >= 0) & (transformed_coords[1, :] < output_shape[0])
    
        # Update valid coordinates only
        output_img[transformed_coords[1, valid_mask], transformed_coords[0, valid_mask]] = \
            img[transformed_coords[1, valid_mask], transformed_coords[0, valid_mask]]
    
        return output_img



    def format_image(self, img):
        img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return img

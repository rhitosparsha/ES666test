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
            original_height, original_width = img.shape[:2]
            new_height = 200
            new_width = int((new_height/original_height)*original_width)
            img = cv2.resize(img, (new_height, new_width))
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
        kp_left, des_left = self.get_keypoints(left_img)
        kp_right, des_right = self.get_keypoints(right_img)

        matched_points = self.get_matched_points(kp_left, des_left, kp_right, des_right)

        if not self.check_order(matched_points):
            left_img, right_img = right_img, left_img
            matched_points[:, [0, 1]] = matched_points[:, [1, 0]]
            transform_left = not transform_left

        homography_matrix = self.ransac(matched_points)
        inverse_homography_matrix = np.linalg.inv(homography_matrix)

        right_image_shape = right_img.shape
        left_image_shape = left_img.shape

        left_image_corners = np.float32([[0, 0], [0, left_image_shape[0]], [left_image_shape[1], left_image_shape[0]], [left_image_shape[1], 0]]).reshape(-1, 1, 2)
        right_image_corners = np.float32([[0, 0], [0, right_image_shape[0]], [right_image_shape[1], right_image_shape[0]], [right_image_shape[1], 0]]).reshape(-1, 1, 2)
        
        if transform_left:
            left_image_corners = self.perspective_transform(left_image_corners, homography_matrix)
        else:
            right_image_corners = self.perspective_transform(right_image_corners, inverse_homography_matrix)
            
        list_of_points = np.concatenate((left_image_corners, right_image_corners), axis=0)

        [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

        if transform_left:
            translation_matrix = (np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])).dot(homography_matrix)
            output_img = self.wrap_perspective(left_img, translation_matrix, (y_max-y_min, x_max-x_min, 3))
            output_img[-y_min:right_image_shape[0]-y_min, -x_min:right_image_shape[1]-x_min] = right_img
            return output_img, homography_matrix, not transform_left
        else:
            translation_matrix = (np.array([[1, 0, 0], [0, 1, -y_min], [0, 0, 1]])).dot(inverse_homography_matrix)
            output_img = self.wrap_perspective(right_img, translation_matrix, (y_max-y_min, x_max-x_min, 3))
            output_img[-y_min:left_image_shape[0]-y_min, :left_image_shape[1]] = left_img
            return output_img, inverse_homography_matrix, not transform_left

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

    def wrap_perspective(self, img, homography_matrix, shape):
        output_img = np.zeros(shape, dtype=np.uint8)
        h, w = img.shape[:2]
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        ones = np.ones_like(x_coords.flatten())
        pixel_coords = np.vstack([x_coords.flatten(), y_coords.flatten(), ones])
        transformed_coords = np.dot(homography_matrix, pixel_coords)
        transformed_coords /= transformed_coords[2, :]
        x_transformed = transformed_coords[0, :].reshape(h, w)
        y_transformed = transformed_coords[1, :].reshape(h, w)

        valid_x = np.isfinite(x_transformed)
        valid_y = np.isfinite(y_transformed)
        valid_coords = valid_x & valid_y
        x_transformed = np.clip(x_transformed, 0, shape[1]-1).astype(np.int32)
        y_transformed = np.clip(y_transformed, 0, shape[0]-1).astype(np.int32)
        output_img[y_transformed[valid_coords], x_transformed[valid_coords]] = img[valid_coords]
        return output_img

    def format_image(self, img):
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        return img[y:y+h, x:x+w]

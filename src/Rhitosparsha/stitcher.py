import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from PIL import Image

class PanaromaStitcher:
    def __init__(self, image_files, focal_length=800, Flag = False):
        self.focal_length = focal_length
        if Flag == True:
            self.images = [self.cylindrical_warp(cv2.imread(img), focal_length) for img in image_files]
        else:
            self.images = [cv2.imread(img) for img in image_files]
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

        # Crop the image to remove black borders
        min_x, max_x = u[valid_mask].min(), u[valid_mask].max()
        min_y, max_y = v[valid_mask].min(), v[valid_mask].max()
        cropped_cylindrical_img = cylindrical_img[min_y:max_y, min_x:max_x]

        return cropped_cylindrical_img


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
        # Custom method for computing homography matrix
        A = []
        for i in range(len(src_pts)):
            x, y = src_pts[i][0]
            xp, yp = dst_pts[i][0]
            A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
            A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
        A = np.array(A)
        U, S, Vh = np.linalg.svd(A)
        L = Vh[-1, :] / Vh[-1, -1]
        H = L.reshape(3, 3)
        return H

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

    def stitch_images(self):
        # Find the center index
        center_index = len(self.images) // 2
        all_homographies = self.all_homographies()

        # Stitch the left side from the first image up to the center
        left_panorama = self.images[0].copy()
        mask_left = single_weights_matrix(left_panorama.shape[:2])

        for i in range(0, center_index):
            H =  all_homographies[i]
            left_panorama, mask_left = self.left_apply_homography(left_panorama, H, self.images[i + 1], mask_left)

        # Stitch the right side from the last image back to the center
        right_panorama = self.images[-1]
        mask_right = single_weights_matrix(right_panorama.shape[:2])

        for i in range(len(self.images) - 2, center_index, -1):
            H_inv = np.linalg.inv(all_homographies[i])
            right_panorama, mask_right = self.right_apply_homography(right_panorama, H_inv, self.images[i], mask_right)

        # Merge left and right panoramas
        final_H = np.linalg.inv(all_homographies[center_index])
        final_panorama, _ = self.combined_apply_homography(right_panorama, left_panorama, final_H, mask_right, mask_left)

        return final_panorama, all_homographies

    def left_apply_homography(self, img, H, base_img, img_mask):
        # Get dimensions and define corners
        h, w = img.shape[:2]
        base_h, base_w = base_img.shape[:2]
        corners_img = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        corners_base = np.float32([[0, 0], [base_w, 0], [base_w, base_h], [0, base_h]]).reshape(-1, 1, 2)
        # Use the last translation matrix or identity if this is the first image
        last_translation = np.linalg.inv(self.translation_left_matrices[-1]) if self.translation_left_matrices else np.eye(3, dtype=np.float32)
        # Transform corners to find canvas bounds
        transformed_corners_img = cv2.perspectiveTransform(corners_img, H @ last_translation)
        all_corners = np.vstack((transformed_corners_img, corners_base))
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())
        width, height = xmax - xmin, ymax - ymin

        # Adjust homography for translation
        translation_matrix = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float32)
        adjusted_H = translation_matrix @ H @ last_translation
        self.translation_left_matrices.append(translation_matrix)
        

        # Create canvas and warp img with blending masks
        warped_img = cv2.warpPerspective(img, adjusted_H, (width, height))
        warped_base = cv2.warpPerspective(base_img, translation_matrix, (width, height))

        # Create and warp the masks
        mask = single_weights_matrix(base_img.shape[:2])
        warped_mask1 = cv2.warpPerspective(img_mask, adjusted_H, (width, height))
        warped_mask2 = cv2.warpPerspective(mask, translation_matrix, (width, height))
        # Normalize the masks
        combined_mask = warped_mask1 + warped_mask2
        normalized_mask1 = np.divide(warped_mask1, combined_mask, where=combined_mask > 0)
        normalized_mask2 = np.divide(warped_mask2, combined_mask, where=combined_mask > 0)

        # Blend images with masks
        blended = (warped_img * normalized_mask1[:, :, None] + warped_base * normalized_mask2[:, :, None]).astype(np.uint8)

        return blended, combined_mask/combined_mask.max()

    def right_apply_homography(self, img, H, base_img, img_mask):
        # Get dimensions and define corners
        h, w = img.shape[:2]
        base_h, base_w = base_img.shape[:2]
        corners_img = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        corners_base = np.float32([[0, 0], [base_w, 0], [base_w, base_h], [0, base_h]]).reshape(-1, 1, 2)
        # Use the last translation matrix or identity if this is the first image
        last_translation = np.linalg.inv(self.translation_right_matrices[-1]) if self.translation_right_matrices else np.eye(3, dtype=np.float32)
        # Transform corners to find canvas bounds
        transformed_corners_img = cv2.perspectiveTransform(corners_img, H @ last_translation)
        all_corners = np.vstack((transformed_corners_img, corners_base))
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())
        width, height = xmax - xmin, ymax - ymin

        # Adjust homography for translation
        translation_matrix = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float32)
        adjusted_H = translation_matrix @ H @ last_translation
        self.translation_right_matrices.append(translation_matrix)
        

        # Create canvas and warp img with blending masks
        warped_img = cv2.warpPerspective(img, adjusted_H, (width, height))
        warped_base = cv2.warpPerspective(base_img, translation_matrix, (width, height))

        # Create and warp the masks
        mask = single_weights_matrix(base_img.shape[:2])
        warped_mask1 = cv2.warpPerspective(img_mask, adjusted_H, (width, height))
        warped_mask2 = cv2.warpPerspective(mask, translation_matrix, (width, height))
        # Normalize the masks
        combined_mask = warped_mask1 + warped_mask2
        normalized_mask1 = np.divide(warped_mask1, combined_mask, where=combined_mask > 0)
        normalized_mask2 = np.divide(warped_mask2, combined_mask, where=combined_mask > 0)

        # Blend images with masks
        blended = (warped_img * normalized_mask1[:, :, None] + warped_base * normalized_mask2[:, :, None]).astype(np.uint8)

        return blended, combined_mask/combined_mask.max()

    def combined_apply_homography(self, right_panorama, left_panorama, final_H, mask_right, mask_left):
        # Get dimensions and define corners
        h, w = right_panorama.shape[:2]
        base_h, base_w = left_panorama.shape[:2]

        corners_right = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        corners_left = np.float32([[0, 0], [base_w, 0], [base_w, base_h], [0, base_h]]).reshape(-1, 1, 2)

        # Use the last translation matrix or identity if this is the first image
        last_translation_left = np.linalg.inv(self.translation_left_matrices[-1])
        last_translation_right = np.linalg.inv(self.translation_right_matrices[-1]) 
        
        # Transform corners to find canvas bounds
        transformed_corners_right = cv2.perspectiveTransform(corners_right, final_H @ last_translation_right)
        transformed_corners_left = cv2.perspectiveTransform(corners_left, last_translation_left)
        all_corners = np.vstack((transformed_corners_right, transformed_corners_left))
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())
        width, height = xmax - xmin, ymax - ymin

        # Adjust homography for translation
        translation_matrix = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float32)
        adjusted_H_right = translation_matrix @ final_H @ last_translation_right
        adjusted_H_left = translation_matrix @ last_translation_left

        # Create canvas and warp img with blending masks
        warped_right = cv2.warpPerspective(right_panorama, adjusted_H_right, (width, height))
        warped_left = cv2.warpPerspective(left_panorama, adjusted_H_left, (width, height))
        
        # Create and warp the masks
        warped_mask1 = cv2.warpPerspective(mask_left, adjusted_H_left, (width, height))
        warped_mask2 = cv2.warpPerspective(mask_right, adjusted_H_right, (width, height))
        
        # Normalize the masks
        combined_mask = warped_mask1 + warped_mask2
        normalized_mask1 = np.divide(warped_mask1, combined_mask, where=combined_mask > 0)
        normalized_mask2 = np.divide(warped_mask2, combined_mask, where=combined_mask > 0)
        
        # Blend images with masks
        blended = (warped_left * normalized_mask1[:, :, None] + warped_right * normalized_mask2[:, :, None]).astype(np.uint8)

        return blended, combined_mask/combined_mask.max()


def single_weights_array(size: int) -> np.ndarray:
    if size % 2 == 1:
        return np.concatenate(
            [np.linspace(0, 1, (size + 1) // 2), np.linspace(1, 0, (size + 1) // 2)[1:]]
        )
    else:
        return np.concatenate([np.linspace(0, 1, size // 2), np.linspace(1, 0, size // 2)])

def single_weights_matrix(shape: tuple[int]) -> np.ndarray:
    return (
        single_weights_array(shape[0])[:, np.newaxis]
        @ single_weights_array(shape[1])[np.newaxis, :]
    )

# Usage
if __name__ == "__main__":
    image_files = sorted(glob.glob('ES666-Assignment3/Images/I6/*'))
    if not image_files:
        raise ValueError("No images found in the specified directory. Check the path and try again.")

    stitcher = PanaromaStitcher(image_files, focal_length=700, Flag=True)  # Adjust focal length if necessary
    final_panorama, homography_matrix_list = stitcher.stitch_images()
    plt.imshow(cv2.cvtColor(final_panorama, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    plt.imsave('Processed_I6.jpg', cv2.cvtColor(final_panorama, cv2.COLOR_BGR2RGB))
    print(homography_matrix_list)

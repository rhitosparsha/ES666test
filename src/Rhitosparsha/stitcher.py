import glob
import cv2
import os
import numpy as np

class PanaromaStitcher():
    def __init__(self):
        pass

    def detect_and_compute(self, img):
        """Detect keypoints and compute descriptors."""
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()  # Create an instance of SIFT
        keypoints, descriptors = sift.detectAndCompute(gray_img, None)
        return keypoints, descriptors

    def compute_homography(self, src_pts, dst_pts):
        """Compute the homography matrix using DLT."""
        A = []
        for i in range(len(src_pts)):
            x, y = src_pts[i][0], src_pts[i][1]
            u, v = dst_pts[i][0], dst_pts[i][1]
            A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
            A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

        A = np.array(A)
        U, S, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        return H / H[-1, -1]  # Normalize

    def warp_perspective(self, src_img, H, dsize):
        """Warp the source image using the given homography matrix."""
        warped_img = np.zeros((dsize[1], dsize[0], 3), dtype=src_img.dtype)
        H_inv = np.linalg.inv(H)

        for y in range(dsize[1]):
            for x in range(dsize[0]):
                dest_coord = np.array([x, y, 1])
                src_coord = H_inv @ dest_coord
                src_coord /= src_coord[2]

                src_x, src_y = int(src_coord[0]), int(src_coord[1])

                # Check bounds
                if 0 <= src_x < src_img.shape[1] and 0 <= src_y < src_img.shape[0]:
                    warped_img[y, x] = src_img[src_y, src_x]

        return warped_img

    def blend_images(self, warped_img, base_img, H):
        """Blend the warped image with the base image using the homography."""
        h1, w1 = warped_img.shape[:2]
        h2, w2 = base_img.shape[:2]
        
        # Create a canvas for the blended image
        blended_height = max(h1, h2)
        blended_width = max(w1, w2)
        blended_image = np.zeros((blended_height, blended_width, 3), dtype=np.uint8)
        blended_image[:h2, :w2] = base_img  # Place the base image

        H_inv = np.linalg.inv(H)

        for y in range(h1):
            for x in range(w1):
                src_coord = np.array([x, y, 1])
                dst_coord = H_inv @ src_coord
                dst_coord /= dst_coord[2]

                dst_x, dst_y = int(dst_coord[0]), int(dst_coord[1])

                # Check bounds
                if 0 <= dst_x < blended_width and 0 <= dst_y < blended_height:
                    warped_pixel = warped_img[y, x]

                    # Only blend if the warped pixel is not black
                    if not np.array_equal(warped_pixel, [0, 0, 0]):
                        base_pixel = blended_image[dst_y, dst_x]
                        alpha = 0.5  # Blend factor
                        blended_image[dst_y, dst_x] = (
                            alpha * warped_pixel + (1 - alpha) * base_pixel
                        )

        return blended_image.astype(np.uint8)

    def make_panaroma_for_images_in(self, path):
        imf = path
        all_images = sorted(glob.glob(imf + os.sep + '*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        # Initialize the list for homography matrices
        homography_matrix_list = []

        # Load the first image to start stitching
        stitched_image = cv2.imread(all_images[0])
        keypoints_base, descriptors_base = self.detect_and_compute(stitched_image)

        # Iterate through the remaining images
        for i in range(1, len(all_images)):
            next_img = cv2.imread(all_images[i])
            keypoints_next, descriptors_next = self.detect_and_compute(next_img)

            # Match descriptors
            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(descriptors_base, descriptors_next, k=2)

            # Apply ratio test
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

            if len(good_matches) > 4:
                # Extract matching points
                src_pts = np.float32([keypoints_base[m.queryIdx].pt for m in good_matches])
                dst_pts = np.float32([keypoints_next[m.trainIdx].pt for m in good_matches])

                # Compute homography
                H = self.compute_homography(src_pts, dst_pts)
                homography_matrix_list.append(H)

                # Warp the next image
                height, width = stitched_image.shape[:2]
                dsize = (width + next_img.shape[1], height)
                warped_image = self.warp_perspective(next_img, H, dsize)

                # Blend the images
                stitched_image = self.blend_images(warped_image, stitched_image, H)

                # Update the base image for the next iteration
                keypoints_base, descriptors_base = keypoints_next, descriptors_next
            else:
                print(f"Not enough matches found between images {i-1} and {i}.")

        return stitched_image, homography_matrix_list


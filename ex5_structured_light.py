import os
import cv2
import numpy as np
import open3d as o3d
from scipy.io import loadmat
from tqdm import tqdm
from lib.visualization.image import show_images


class Camera:
    def __init__(self, t, r, k):
        """
        Creates a Camera instance

        Parameters
        ----------
        t (ndarray): Translation vector. Shape (3, 1)
        r (ndarray): Rotation matrix. Shape (3, 3)
        k (ndarray): Camera matrix. Shape (3, 3)
        """
        self.t = t
        self.r = r
        self.k = k
        self.c = np.dot(np.linalg.inv(r), t) * -1
        self.proj = np.matmul(k, np.hstack([r, t]))

    def add_rect_proj(self, H, x_dim, y_dim):
        """
        Add the rectified projection matrix

        Parameters
        ----------
        H (ndarray): Homography matrix. Shape (3, 3)
        x_dim (ndarray): The x dimension. Shape (2)
        y_dim (ndarray): The y dimension. Shape (2)
        """
        self.H = H
        self.rect_proj = np.dot(H, self.proj)
        self.out_dim = (int(x_dim[1]), int(y_dim[1]))

    def warp_img(self, img):
        """
        Applies the rectified transformation to the image.

        Parameters
        ----------
        img (nadrray): The image. Shape (height, width)

        Returns
        -------
        img_rect (nadrray): The rectified image. Shape (rect_height, rect_width)
        """
        img_rect = cv2.warpPerspective(img, self.H, self.out_dim, flags=cv2.INTER_NEAREST)
        return img_rect


class StereoCameras:
    def __init__(self, t_r, r_r, k_r, t_l, r_l, k_l):
        """
        Creates a StereoCameras instance

        Parameters
        ----------
        t_r (ndarray): Translation vector right. Shape (3, 1)
        r_r (ndarray): Rotation matrix right. Shape (3, 3)
        k_r (ndarray): Camera matrix right. Shape (3, 3)
        t_l (ndarray): Translation vector left. Shape (3, 1)
        r_l (ndarray): Rotation matrix left. Shape (3, 3)
        k_l (ndarray): Camera matrix left. Shape (3, 3)
        """
        self.cam_l = Camera(t_l, r_l, k_l)
        self.cam_r = Camera(t_r, r_r, k_r)
        self.vec_a = np.squeeze(self.cam_r.c - self.cam_l.c)  # Vector between camera centers
        self.vec_b = (self.cam_l.r[2, :] + self.cam_r.r[2, :]).T / 2  # Mean optical axis
        self.vec_c = np.cross(self.vec_b, self.vec_a)
        self.vec_a /= np.linalg.norm(self.vec_a)
        self.vec_b /= np.linalg.norm(self.vec_b)
        self.vec_c /= np.linalg.norm(self.vec_c)

    @staticmethod
    def form_hbas(va, vb, vc):
        Hbas = np.hstack([np.expand_dims(va, axis=1), np.expand_dims(vb, axis=1), np.expand_dims(vc, axis=1)])
        Hbas = np.vstack([Hbas, [0, 0, 1]])
        return Hbas

    @staticmethod
    def form_homography(proj, hbas):
        H = np.linalg.inv(np.dot(proj, hbas))
        return H / H[2, 2]

    @staticmethod
    def make_k(x, y):
        k = np.eye(3)
        k[0, 2] = x
        k[1, 2] = y
        return k

    def calc_xy_data(self, h_l, h_r, shape):
        Q = np.ones((3, 4))
        Q[0, 1], Q[0, 3], Q[1, 2], Q[1, 3] = shape[1], shape[1], shape[0], shape[0]
        q = np.hstack([np.dot(h_l, Q), np.dot(h_r, Q)])
        q /= q[2]
        out_x_dim = np.round([np.min(q[0]), np.max(q[0])])
        out_y_dim = np.round([np.min(q[1]), np.max(q[1])])
        dx = out_x_dim[0] - 1
        dy = out_y_dim[0] - 1
        out_x_dim -= dx
        out_y_dim -= dy
        k = self.make_k(-dx, -dy)
        h_l = np.dot(k, h_l)
        h_r = np.dot(k, h_r)
        return h_l, h_r, out_x_dim, out_y_dim

    def triangulate_point(self, q1, q2, use_rectified=False):
        """
        Triangulates the given point

        Parameters
        ----------
        q1 (ndarray): . Shape(2)
        q2 (ndarray): . Shape(2)
        use_rectified (bool): If True the rectified projection matrix will be used

        Returns
        -------
        Q (ndarray): . Shape(3)
        """

        p_l = self.cam_l.proj if not use_rectified else self.cam_l.rect_proj
        p_r = self.cam_r.proj if not use_rectified else self.cam_r.rect_proj

        b = np.zeros((4, 4))
        b[0, :] = p_l[0, :] - q1[0] * p_l[2, :]
        b[1, :] = p_l[1, :] - q1[1] * p_l[2, :]
        b[2, :] = p_r[0, :] - q2[0] * p_r[2, :]
        b[3, :] = p_r[1, :] - q2[1] * p_r[2, :]
        _, _, v = np.linalg.svd(b)
        Q = v.T[:3, 3] / v.T[3, 3]
        return Q

    def calc_rect_homographies(self, shape, upscale=1.2):
        """
        Calculate the rectified homographies
        """
        img_c = np.asarray(shape) / 2
        b = np.zeros((4, 4))
        V = self.triangulate_point(img_c, img_c)

        Hbas = self.form_hbas(self.vec_a, self.vec_c, V)
        H_l = self.form_homography(self.cam_l.proj, Hbas)
        H_r = self.form_homography(self.cam_r.proj, Hbas)
        scale = np.max([np.sqrt(np.linalg.det(H_l[:2, :2])), np.sqrt(np.linalg.det(H_r[:2, :2]))]) / upscale

        Hbas = self.form_hbas(scale * self.vec_a, scale * self.vec_c, V)
        k = self.make_k(-img_c[0] * upscale, -img_c[1] * upscale)
        Hbas[:3, :] = np.dot(Hbas[:3, :], k)
        H_l = self.form_homography(self.cam_l.proj, Hbas)
        H_r = self.form_homography(self.cam_r.proj, Hbas)

        H_l, H_r, out_x_dim, out_y_dim = self.calc_xy_data(H_l, H_r, shape)
        self.cam_l.add_rect_proj(H_l, out_x_dim, out_y_dim)
        self.cam_r.add_rect_proj(H_r, out_x_dim, out_y_dim)


def load_cam_params(param_file):
    """
    Loads the camera parameters

    Parameters
    ----------
    param_file (str): The file path

    Returns
    -------
    stereoCameras (StereoCameras): The stereo cameras
    """
    cam_params = loadmat(param_file)
    t_l, r_l, k_l = cam_params['T1'], cam_params['R1'], cam_params['A1']
    t_r, r_r, k_r = cam_params['T2'], cam_params['R2'], cam_params['A2']
    stereoCameras = StereoCameras(t_r, r_r, k_r, t_l, r_l, k_l)
    return stereoCameras


def load_image_data(data_file):
    """
    Loads the image data

    Parameters
    ----------
    data_file (str): The file path

    Returns
    -------
    data_list (list): List there contains the Black_Left, White_Left, Black_Right, White_Right
                      Pos_Left, Neg_Left, Pos_Right, Neg_Right images. Shape (8)
    """
    data = loadmat(data_file)
    data_list = [data['Black_Left'], data['White_Left'], data['Black_Right'], data['White_Right']]
    keys = ['Pos_Left', 'Neg_Left', 'Pos_Right', 'Neg_Right']
    data_list.extend([np.array(np.squeeze(data[key]).tolist()) for key in keys])
    return data_list


def get_occlusion_mask(black, white):
    """
    Creates the occlusion mask

    Parameters
    ----------
    black (ndarray): . Shape (height, width)
    white (ndarray): . Shape (height, width)

    Returns
    -------
    occ_mask (ndarray): . Shape (height, width)
    """
    diff = np.abs(white - black)
    mask = diff > 0.5 * np.max(white)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((4, 4)))
    occ_mask = np.zeros(mask.shape)
    occ_mask[4:-4, 4:-4] = mask[4:-4, 4:-4]
    return occ_mask


def get_encoding_image(pos, neg, mask):
    """
    Creates the encoding image

    Parameters
    ----------
    pos (ndarray): The positive image. Shape (n, height, width)
    neg (ndarray): The negative image. Shape (n, height, width)
    mask (ndarray): The occlusion mask. Shape (height, width)

    Returns
    -------
    enc_img (ndarray): The encoding image. Shape (height, width)
    """
    diff_img = np.zeros(pos.shape)
    n_img = pos.shape[0]
    for i in range(n_img):
        diff = pos[i, :, :] - neg[i, :, :]
        diff[mask == 0] = 0
        diff_img[i, :, :] = cv2.GaussianBlur(diff, (5, 5), 0)

    enc_img = np.zeros(pos.shape[-2:])
    for i in range(n_img):
        enc_img = enc_img + 2 ** i * (diff_img[i, :, :] > 0)
    return enc_img


def scan_encoding(enc_l, enc_r):
    """
    Scans the rectified encoding images

    Parameters
    ----------
    enc_l (ndarray): The left rectified encoding image. Shape (height, width)
    enc_r (ndarray): The right rectified encoding image. Shape (height, width)

    Returns
    -------
    matchs (list): A list with the matchs. Shape (n, 3)
    """
    def sort_rows(a):
        """
        Matlabs sortrows function works similarly to the provided function

        Parameters
        ----------
        a (ndarray): The matrix to sort

        Returns
        -------
        a (ndarray): The sorted matrix. Sorted an axis 2 (unstable), 1 (stable) and then 0 (stable)
        """
        a = a[a[:, 2].argsort()]  # First sort doesn't need to be stable.
        a = a[a[:, 1].argsort(kind='mergesort')]
        return a[a[:, 0].argsort(kind='mergesort')]

    def find_edges(enc, scan_line):
        # Find the differences between neighboring pixels on the scan line
        diffs = enc[scan_line, 1:] - enc[scan_line, :-1]
        edges = []

        if np.any(diffs != 0):
            # Get the column where there are neighboring pixels with differences
            cols = np.where(diffs != 0.)[0]
            # Create the edges list
            label1 = enc[scan_line, cols].astype(np.uint8)
            label2 = enc[scan_line, cols + 1].astype(np.uint8)
            edges = np.array([label1, label2, cols]).T
        return edges

    def remove_doubles(edges):
        no_doubles = [edges[0]]
        old_label1, old_label2, _ = edges[0]

        for (label1, label2, col) in edges[1:]:
            if label1 == old_label1 and label2 == old_label2:
                continue

            old_label1 = label1
            old_label2 = label2
            no_doubles.append([label1, label2, col])

        no_doubles = np.array(no_doubles)
        return no_doubles

    def compare_edge(edge_i, edge_j):
        temp1 = (edge_i[0] * 2 ** 8 + edge_j[1])
        temp2 = (edge_i[0] * 2 ** 8 + edge_j[1])
        return temp1 > temp2

    # Get the number of scan lines
    scan_lines = enc_l.shape[0]

    matches = []
    for scan_line in tqdm(range(scan_lines)):
        # Find the edges in both rectified encoding images
        edges_l = find_edges(enc_l, scan_line)
        edges_r = find_edges(enc_r, scan_line)

        if len(edges_l) == 0 or len(edges_r) == 0:
            continue

        # Sort and remove doubles from the edges
        edges_l = sort_rows(edges_l)
        edges_l = remove_doubles(edges_l)
        edges_r = sort_rows(edges_r)
        edges_r = remove_doubles(edges_r)

        # Find the matches
        i = 1
        j = 1
        while i < len(edges_r) and j < len(edges_l):
            label1_l, label2_l, col_l = edges_l[j]
            label1_r, label2_r, col_r = edges_r[i]

            if label1_l == label1_r and label2_l == label2_r:
                matches.append([scan_line, col_l, col_r])
                i += 1
                j += 1
            elif compare_edge(edges_r[i], edges_l[j]):
                j += 1
            else:
                i += 1

    return matches


def triangulate_matches(cams, matches):
    """
    Triangulates the matches

    Parameters
    ----------
    cams (StereoCameras): The stereo cameras
    matches (ndarray): The matches. Shape (n,3)

    Returns
    -------
    Qs (ndarray): The triangulated 3D points. Shape (n, 3)
    """
    Qs = []
    for match in matches:
        Qs.append(cams.triangulate_point((match[0], match[1]), (match[0], match[2]), use_rectified=True))
    return np.array(Qs)


def visualize_points(points):
    """
    Visualizes the 3D points

    Parameters
    ----------
    points (ndarray): The 3D points. Shape (n, 3)
    """
    # Use Open3D to visualize the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])


def show_matches(img_l, img_r, matches):
    img_l_draw = img_l.copy()
    img_l_draw = cv2.cvtColor(img_l_draw.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    img_r_draw = img_r.copy()
    img_r_draw = cv2.cvtColor(img_r_draw.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    random_color = lambda: tuple([int(c) for c in np.random.random(size=3) * 256])

    for s_line, col_l, col_r in matches:
        color = random_color()
        cv2.circle(img_l_draw, (col_l, s_line), 1, color, -1)
        cv2.circle(img_r_draw, (col_r, s_line), 1, color, -1)

    cv2.imshow("Matches left", img_l_draw)
    cv2.imshow("Matches right", img_r_draw)
    cv2.waitKey(0)


def main():
    # Load the data
    data_dir = '../data/'
    cams = load_cam_params(os.path.join(data_dir, "CameraData.mat"))
    black_l, white_l, black_r, white_r, pos_l, neg_l, pos_r, neg_r = load_image_data(
        os.path.join(data_dir, "StrlImageData.mat"))
    # Show the images
    show_images(neg_l)
    # Calculate the rectified homographies
    cams.calc_rect_homographies(black_l.shape)

    # Get the occlusion masks
    occ_l = get_occlusion_mask(black_l, white_l)
    occ_r = get_occlusion_mask(black_r, white_r)
    show_images(occ_l * 255, image_title="Occlusion masks")

    # Get the encoding images
    enc_l = get_encoding_image(pos_l, neg_l, occ_l)
    enc_r = get_encoding_image(pos_r, neg_r, occ_r)
    show_images(enc_l, image_title="Encoding image")

    # Apply the rectified transformation on the occlusion masks
    occ_l_rect = cams.cam_l.warp_img(occ_l)
    occ_r_rect = cams.cam_r.warp_img(occ_r)

    # Apply the rectified transformation on the encoding images
    enc_l_rect = cams.cam_l.warp_img(enc_l)
    enc_r_rect = cams.cam_r.warp_img(enc_r)

    # Apply the occlusion masks to the encoding images
    enc_l_rect[occ_l_rect == 0] = 0
    enc_r_rect[occ_r_rect == 0] = 0
    show_images(enc_l_rect, image_title="Masked Rectified encoding image (left)")
    show_images(enc_r_rect, image_title="Masked Rectified encoding image (right)")

    # Find the matches
    matches = scan_encoding(enc_l_rect, enc_r_rect)
    show_matches(enc_l_rect, enc_r_rect, matches)
    cv2.destroyAllWindows()

    # Triangulate the matches
    points = triangulate_matches(cams, matches)
    # Visualize the points
    visualize_points(points)


if __name__ == "__main__":
    main()

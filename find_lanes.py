import pickle
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class WarpImage():
    def __init__(self):
        with open('mtx.pkl', 'rb') as f:
            self.mtx = pickle.load(f)
        with open('dist.pkl', 'rb') as f:
            self.dist = pickle.load(f)
        with open('perspective_mtx.pkl', 'rb') as f:
            self.perspective_mtx = pickle.load(f)

        self.perspective_mtx_inv = np.linalg.inv(self.perspective_mtx)

    def warp(self, image):
        image_undistorted = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        return cv2.warpPerspective(image_undistorted, self.perspective_mtx, image.shape[1::-1])

    def restore(self, warped_image):
        return cv2.warpPerspective(warped_image, self.perspective_mtx_inv, warped_image.shape[1::-1])


class ColorProcessor():
    def execute(self, img, s_th_min=167, s_th_max=255):
        img = np.copy(img)
        # Convert to HLS color space and separate the S channel
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        s_channel = hsv[:, :, 2]
        color_binary = np.zeros_like(s_channel)
        color_binary[(s_th_min <= s_channel) & (s_th_max >= s_channel)] = 1
        return color_binary


class SobelProcessor():
    @staticmethod
    def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return binary_output

    @staticmethod
    def mag_thresh(img, thresh=(0, 255)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        s_binary = np.zeros_like(gray)
        s_binary[(gradmag > thresh[0]) & (gradmag <= thresh[1])] = 1
        return s_binary

    @staticmethod
    def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
        return binary_output

    def execute(self, image,
                gx_th_min=31, gx_th_max=80,
                gy_th_min=41, gy_th_max=112,
                mg_th_min=37, mg_th_max=119,
                dir_kernel_size=15, dir_th_min=0.7, dir_th_max=1.3):
        gradx = self.abs_sobel_thresh(image, orient='x', thresh=(gx_th_min, gx_th_max))
        grady = self.abs_sobel_thresh(image, orient='y', thresh=(gy_th_min, gy_th_max))
        mag_binary = self.mag_thresh(image, thresh=(mg_th_min, mg_th_max))
        dir_binary = self.dir_threshold(image, sobel_kernel=dir_kernel_size, thresh=(dir_th_min, dir_th_max))
        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        return combined


class ImageProcessor():
    def __init__(self):
        self.color_proc = ColorProcessor()
        self.sobel_proc = SobelProcessor()

    def execute(self, image):
        image = np.copy(image)
        cimage = self.color_proc.execute(image)
        simage = self.sobel_proc.execute(image)
        combine = np.zeros_like(cimage)
        combine[(cimage == 1) | (simage == 1)] = 1
        return combine


class Pipeline():
    def __init__(self):
        self.image_prc = ImageProcessor()
        self.warp = WarpImage()

    def execute(self, image):
        image = self.image_prc.execute(image)
        image = self.warp.warp(image)
        return image


# class Line():
#     def __init__(self):
#         # was the line detected in the last iteration?
#         self.detected = False
#         # x values of the last n fits of the line
#         self.recent_xfitted = []
#         # average x values of the fitted line over the last n iterations
#         self.bestx = None
#         # polynomial coefficients averaged over the last n iterations
#         self.best_fit = None
#         # polynomial coefficients for the most recent fit
#         self.current_fit = [np.array([False])]
#         # radius of curvature of the line in some units
#         self.radius_of_curvature = None
#         # distance in meters of vehicle center from the line
#         self.line_base_pos = None
#         # difference in fit coefficients between last and new fits
#         self.diffs = np.array([0, 0, 0], dtype='float')
#         # x values for detected line pixels
#         self.allx = None
#         # y values for detected line pixels
#         self.ally = None


from enum import Enum

class LineType(Enum):
    left = 1
    right = 2

class Line():
    xm_per_pix = 3.7 / 700
    ym_per_pix = 30 / 720

    def __init__(self, shape, line_type):
        self.line_type = line_type
        self.shape = shape
        self.current_x = []
        self.current_y = []

    def fit_in_pixel(self):
        return np.polyfit(self.current_y, self.current_x, 2)

    def fit_in_meter(self):
        return np.polyfit(self.current_y * Line.ym_per_pix, self.current_x * Line.xm_per_pix, 2)

    def get_x_in_pixel(self, y_in_pixel):
        coef = self.fit_in_pixel()
        return coef[0]*y_in_pixel**2 + coef[1]*y_in_pixel + coef[2]

    def line_position_in_pixel(self):
        ploty = np.linspace(0, self.shape[0] - 1, self.shape[0])
        x = self.get_x_in_pixel(ploty)
        if self.line_type is LineType.left:
            return np.array([np.transpose(np.vstack([x, ploty]))])
        else:
            return np.array([np.flipud(np.transpose(np.vstack([x, ploty])))])

    def curvature_in_meter(self):
        y_eval = (self.shape[0] - 1)*Line.ym_per_pix
        coef = self.fit_in_meter()
        return ((1 + (2 * coef[0] * y_eval + coef[1]) ** 2) ** 1.5) / np.absolute(2 * coef[0])

class Window():
    def __init__(self, shape, base_x, nonzeroy, nonzerox):
        self.shape = shape
        self.height = self.shape[1] // 9
        self.margin = 50
        self.minpix = 50
        self.current_base_x = base_x
        self.nonzeroy = nonzeroy
        self.nonzerox = nonzerox
        self.lane_inds = []

    def update(self, index):
        self.__update_position(index)
        self.__collect_indices()

    def __update_position(self, index):
        self.y_low = self.shape[0] - (index + 1) * self.height
        self.y_high = self.shape[0] - index * self.height
        self.x_low = self.current_base_x - self.margin
        self.x_high = self.current_base_x + self.margin

    def __collect_indices(self):
        good_inds = ((self.nonzeroy >= self.y_low) & (self.nonzeroy < self.y_high)
                     & (self.nonzerox >= self.x_low) & (self.nonzerox < self.x_high)).nonzero()[0]
        if np.sum(good_inds) > 0:
            mean = np.mean(self.nonzerox[good_inds])
            if (len(good_inds) > self.minpix) and (np.fabs(mean - self.current_base_x) < 40):
                # Append these indices to the lists
                self.lane_inds.append(good_inds)
                self.current_base_x = np.int(mean)

# def fit_x(coef, y):
#     return coef[0]*y**2 + coef[1]*y + coef[2]
#
# def fit(ploty, x, ym_per_pix=30/720, xm_per_pix=3.7/700):
#     return np.polyfit(ploty*ym_per_pix, x*xm_per_pix, 2)

class LineManager():
    def __init__(self, shape):
        self.shape = shape
        self.left_line = Line(shape, LineType.left)
        self.right_line = Line(shape, LineType.right)

    def execute(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        midpoint = np.int(histogram.shape[0] / 2)

        leftx_base = np.argmax(histogram[:midpoint])  # 左側ピーク
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint  # 右側ピーク

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()  # np.transpose(nonzero)でインデックスの組み合わせを得る事が出来る
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_window = Window(binary_warped.shape, leftx_base, nonzeroy, nonzerox)
        right_window = Window(binary_warped.shape, rightx_base, nonzeroy, nonzerox)

        nwindows = 9
        for window in range(nwindows):
            left_window.update(window)
            right_window.update(window)

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_window.lane_inds)
        right_lane_inds = np.concatenate(right_window.lane_inds)

        self.left_line.current_x = nonzerox[left_lane_inds]
        self.left_line.current_y = nonzeroy[left_lane_inds]
        self.right_line.current_x = nonzerox[right_lane_inds]
        self.right_line.current_y = nonzeroy[right_lane_inds]


    def lane_region(self):
        color_warp = np.zeros(self.shape).astype(np.uint8)

        pts_left = self.left_line.line_position_in_pixel()
        pts_right = self.right_line.line_position_in_pixel()
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        return color_warp

    def curvature(self):
        left = self.left_line.curvature_in_meter()
        right = self.right_line.curvature_in_meter()
        return (left + right)/2

    def center_offset(self):
        camera_x = self.shape[1]/2
        left = self.left_line.get_x_in_pixel(self.shape[0]-1)
        right = self.right_line.get_x_in_pixel(self.shape[0] - 1)
        return ((left + right)/2 - camera_x)*Line.xm_per_pix


pipeline = Pipeline()
warp_image = WarpImage()

def process_image(image):
    img = pipeline.execute(image)
    line_manager = LineManager(image.shape)
    line_manager.execute(img)

    color_warp = line_manager.lane_region()
    color_image = warp_image.restore(color_warp)
    curvature = line_manager.curvature()
    center_offset = line_manager.center_offset()
    cv2.putText(image, "curvature: {0:.2f}[m]".format(curvature), (50, 80), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 255), 3)
    cv2.putText(image, "center offset: {0:.2f}[m]".format(center_offset), (50, 150), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 255), 3)
    result = cv2.addWeighted(image, 1, color_image, 0.3, 0)
    return result


def one_image():
    img_test5 = mpimg.imread("./test_images/test5.jpg")
    result = process_image(img_test5)
    plt.imshow(result)
    plt.show()

def video():
    from moviepy.editor import VideoFileClip
    white_output = 'project_video_output.mp4'
    clip1 = VideoFileClip('project_video.mp4')
    white_clip = clip1.fl_image(process_image).subclip(0,5)
    white_clip.write_videofile(white_output, audio=False)



if __name__ == '__main__':
    video()
    # one_image()
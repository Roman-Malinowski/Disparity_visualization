import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv


class SnaptoCursor(object):
    def __init__(self, ax, x_data, y_data, x_ref, disp_range):
        self.ax = ax
        self.ly = ax.axvline(color='k', alpha=0.2)  # the vert line
        self.marker, = ax.plot([0], [0], marker="o", color="crimson", zorder=3)
        self.x_data = x_data
        self.y_data = y_data
        self.txt = ax.text(0.7, 0.9, '')
        self.disp_range = disp_range
        # x_global is the x cursor position
        self.x_global = x_ref

    def mouse_move(self, event):
        if not event.inaxes is self.ax: return
        # We round xdata for UX
        x, y = round(event.xdata - 1), event.ydata
        indx = np.searchsorted(self.x_data, [x], side='right')[0]
        indx = np.max([0, indx])
        indx = np.min([self.disp_range[1]-self.disp_range[0]-1, indx])
        x = self.x_data[indx]
        y = self.y_data[indx]
        self.ly.set_xdata(x)
        self.marker.set_data([x], [y])
        self.txt.set_text('Disp=%1.0f, Cost=%1.2f' % (indx+disparity_range[0], y))
        self.txt.set_position((x, y))
        self.ax.figure.canvas.draw_idle()
        self.x_global = x


class FullImage(object):
    def __init__(self, ax, x, y, img, padd=[0, 0], title=""):
        self.ax = ax
        self.x = x
        self.y = y
        self.img = img
        self.padd = padd

        self.ax.imshow(self.img, vmin=0.0, vmax=255.)

        # Create a small patch on the pixel
        self.ax.add_patch(self.create_rectangle())
        self.ax.set_title(title)
        self.ax.set_xticks(ticks=[])
        self.ax.set_yticks(ticks=[])

    def create_rectangle(self):
        return patches.Rectangle((self.x, self.y), self.padd[0], self.padd[1], color="r")

    def update_xy_ref(self, event):
        if event.inaxes is not self.ax: return
        x, y = event.xdata, event.ydata

        # Changing the x and y coordinates but not allowing them to be in padding areas
        x = np.min([self.img.shape[1] - 1 - self.padd[1], x])
        self.x = np.max([self.padd[1], x])
        y = np.min([self.img.shape[0] - 1 - self.padd[0], y])
        self.y = np.max([self.padd[0], y])

        # Removing old rectangle
        self.ax.patches.pop()
        self.ax.add_patch(self.create_rectangle())
        self.ax.figure.canvas.draw_idle()


class ImageIcon(object):
    def __init__(self, ax, x, y, curs, img, disp_range, padd=[0, 0], title=""):
        self.ax = ax
        self.x = x
        self.y = y
        self.cursor = curs
        self.img = img
        self.padd = padd
        self.disp = disp_range

        # The processed image already has some padding, so we need to take that into account
        self.ax.imshow(self.img[self.x: self.x + 2 * self.padd[1] + 1,
                                self.y: self.y + 2*self.padd[0] + 1], vmin=0.0, vmax=255.)
        self.ax.set_title(title)
        self.ax.set_xticks(ticks=[])
        self.ax.set_yticks(ticks=[])

    def mouse_move(self, event):
        if event.inaxes is not self.cursor.ax: return
        y = self.y + self.disp[0] + self.cursor.x_global
        # The processed image already has some padding, so we need to take that into account
        self.ax.imshow(self.img[self.x: self.x + 2*self.padd[1] + 1,
                                y: y + 2*self.padd[0] + 1], vmin=0.0, vmax=255.)
        self.ax.figure.canvas.draw_idle()


class ImageBand(object):
    def __init__(self, ax, x, y, curs, img, disp_range, padd=[0, 0], title=""):
        self.ax = ax
        self.x = x
        self.y = y
        self.cursor = curs
        self.img = img
        self.padd = padd
        self.disp = disp_range
        # The processed image already has some padding, so we need to take that into account
        self.img = img[self.x: self.x + 2*self.padd[1] + 1,
                       self.y + self.disp[0]: self.y + self.disp[1] + 2*self.padd[0] + 1]

        # Left and right vertical lines
        self.ly = ax.axvline(color='r', alpha=0.8)
        self.ry = ax.axvline(color='r', alpha=0.8)

        self.ax.imshow(self.img, vmin=0.0, vmax=255.)
        self.ax.set_xticks(ticks=[])
        self.ax.set_yticks(ticks=[])
        self.ax.set_title(title)

    def mouse_move(self, event):
        if not event.inaxes is self.cursor.ax:return
        # The processed image already has some padding, so we need to take that into account
        self.ly.set_xdata(self.cursor.x_global - 0.5)
        self.ry.set_xdata(self.cursor.x_global + 2*self.padd[0] + 0.5)
        self.ax.figure.canvas.draw_idle()


def plot_costs(list_costs, x, y, list_labels=None):
    """
    A function to plot the cost curves for multiple window aggregation for zncc
    Args:
        list_costs: A list of np.array cost volumes
        x: the x coordinate of the left matching pixel
        y: the y coordinate of the left matching pixel
        If set to None, it will be 0:cost.shape[2]
        list_labels: The list of labels for each curve. Default to "window 0", "window 1" etc...

    Returns:

    """
    if list_labels is None:
        list_labels = ["window %s" % i for i in np.arange(1, len(list_costs)+1)]

    ax_ = plt.subplot(311)
    x_axis = np.arange(0, list_costs[0].shape[2])
    min_y = max_y = np.nan
    for cost, label in zip(list_costs, list_labels):
        ax_.plot(x_axis, cost[x, y, :], label=label, linestyle="-.")
        min_y = np.nanmin(np.hstack([min_y, cost[x, y, :]]))
        max_y = np.nanmax(np.hstack([max_y, cost[x, y, :]]))

    margin = 0.05*(max_y-min_y)
    ax_.set_ylim(min_y-margin, max_y+margin)
    ax_.legend()
    ax_.set_title("Cost curve")

    x_tick_coordinates = np.array(ax_.get_xticks())
    ax_.set_xticks(ticks=x_tick_coordinates, labels=(x_tick_coordinates+disparity_range[0]).astype(int))
    return ax_, x_axis, list_costs[0][x, y, :]


def prepare_image(img_path, padd):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.copyMakeBorder(img, padd[1], padd[1], padd[0], padd[0], cv.BORDER_REFLECT)
    return img


if __name__ == "__main__":
    list_costs = [np.load("./cost_volume_w_1.npy"),
                  np.load("./cost_volume_w_2.npy"),
                  np.load("./cost_volume_w_3.npy"),
                  np.load("./cost_volume_w_4.npy"),
                  np.load("./cost_volume_w_5.npy")]

    left_image_path = "./left.png"
    right_image_path = "./right.png"

    X_ref = 300
    Y_ref = 300
    padding = [2, 2]
    disparity_range = [-60, 0]

    left_image = prepare_image(left_image_path, padding)
    right_image = prepare_image(right_image_path, padding)

    """Creating the plot with cost curve graph"""
    plt.figure(figsize=[16., 7.])
    ax_1, x_data, y_data = plot_costs(list_costs, x=X_ref, y=Y_ref)

    # Adding a cursor to the cost curve axis
    cursor = SnaptoCursor(ax_1, x_data, y_data, X_ref, disparity_range)
    cid_curve = plt.connect('motion_notify_event', cursor.mouse_move)

    """Adding the full Left image"""
    ax_full_image = plt.subplot(337)
    full_image = FullImage(ax_full_image, X_ref, Y_ref, left_image, padd=padding, title="Full Left Image")
    cid_full_image = plt.connect('button_press_event', full_image.update_xy_ref)

    """Adding left and right images"""
    ax_left = plt.subplot(338)
    ax_right = plt.subplot(339)

    left_image_patch = ImageIcon(ax_left, X_ref, Y_ref, cursor, left_image,
                                 disparity_range, padd=padding, title="Left patch")
    right_image_patch = ImageIcon(ax_right, X_ref, Y_ref, cursor, right_image,
                                  disparity_range, padd=padding, title="Right patch")

    # Only updating the right image patch
    cid_patch = plt.connect('motion_notify_event', right_image_patch.mouse_move)

    """Adding the right image band"""
    ax_band = plt.subplot(312)
    image_band = ImageBand(ax_band, X_ref, Y_ref, cursor, right_image,
                           disparity_range, padd=padding, title="Right Image on disparity interval")
    cid_band = plt.connect('motion_notify_event', image_band.mouse_move)

    plt.show()
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


class SnaptoCursor(object):
    def __init__(self, ax, x, y, x_ref, disp_range):
        self.ax = ax
        self.ly = ax.axvline(color='k', alpha=0.2)  # the vert line
        self.marker, = ax.plot([0], [0], marker="o", color="crimson", zorder=3)
        self.x = x
        self.y = y
        self.txt = ax.text(0.7, 0.9, '')
        self.disp_range = disp_range
        # x_global is the x cursor position
        self.x_global = x_ref

    def mouse_move(self, event):
        if not event.inaxes is self.ax: return
        # We round xdata for UX
        x, y = round(event.xdata - 1), event.ydata
        indx = np.searchsorted(self.x, [x], side='right')[0]
        indx = np.max([0, indx])
        indx = np.min([self.disp_range[1]-self.disp_range[0]-1, indx])
        x = self.x[indx]
        y = self.y[indx]
        self.ly.set_xdata(x)
        self.marker.set_data([x], [y])
        self.txt.set_text('Disp=%1.0f, Cost=%1.2f' % (indx+disparity_range[0], y))
        self.txt.set_position((x, y))
        self.ax.figure.canvas.draw_idle()
        self.x_global = x


class ImageIcon(object):
    def __init__(self, ax, x, y, curs, img, disp_range, padd=[0, 0], title=""):
        self.ax = ax
        self.x = x
        self.y = y
        self.cursor = curs
        self.img = img
        self.padd = padd
        self.disp = disp_range

        self.ax.set_title(title)
        # The processed image already has some padding, so we need to take that into account
        self.ax.imshow(self.img[self.x: self.x + 2 * self.padd[1] + 1,
                                self.y: self.y + 2*self.padd[0] + 1], vmin=0.0, vmax=255.)
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

        self.ax.set_title(title)
        self.ax.imshow(self.img, vmin=0.0, vmax=255.)
        self.ax.set_xticks(ticks=[])
        self.ax.set_yticks(ticks=[])

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
    ax_.set_xticks(ticks=[0, list_costs[0].shape[2]], labels=[disparity_range[0], disparity_range[1]])
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

    plt.figure(figsize=[16., 7.])
    ax_1, x_data, y_data = plot_costs(list_costs, x=X_ref, y=Y_ref)

    # Adding a cursor to the volume curve axis
    cursor = SnaptoCursor(ax_1, x_data, y_data, X_ref, disparity_range)
    cid_1 = plt.connect('motion_notify_event', cursor.mouse_move)

    # Adding left and right images
    ax_2 = plt.subplot(323)
    ax_3 = plt.subplot(324)

    left_image_icon = ImageIcon(ax_2, X_ref, Y_ref, cursor, left_image, disparity_range, padd=padding, title="Left Image")
    right_image_icon = ImageIcon(ax_3, X_ref, Y_ref, cursor, right_image, disparity_range, padd=padding, title="Right Image")

    # Only updating the right image
    cid_2 = plt.connect('motion_notify_event', right_image_icon.mouse_move)

    ax_4 = plt.subplot(313)
    image_band = ImageBand(ax_4, X_ref, Y_ref, cursor, right_image, disparity_range, padd=padding, title="Image on disparity interval")
    cid_3 = plt.connect('motion_notify_event', image_band.mouse_move)

    plt.show()
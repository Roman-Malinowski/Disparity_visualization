import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv


class CostCurveAndCursor(object):

    def __init__(self, ax, list_cost_volume, x_ref, y_ref, disp_range, padd, list_labels=None):
        """
        An object to create a cursor attached to a cost volume curve
        Args:
            ax: the pyplot Axis object on witch the cursor appears
            list_cost_volume: An array containing the cost volumes for all windows
            x_ref: the x coordinates of the reference point in the left image
            y_ref: the y coordinates of the reference point in the left image
            disp_range: the disparity range of the cost volume
            padd: [horizontal padding, vertical padding]
            list_labels: a list of labels for the legend of the plot
        """
        self.ax = ax
        self.list_cost_volume = list_cost_volume
        self.x_ref = x_ref
        self.y_ref = y_ref

        self.x_data = np.arange(0, list_cost_volume[0].shape[2])
        self.y_data = list_costs[0][x_ref, y_ref, :]
        # x_cursor is the x cursor position
        self.x_cursor = x_ref

        self.disp_range = disp_range
        self.padd = padd
        self.list_labels = list_labels

        self.ly = ax.axvline(color='k', alpha=0.5)  # the vert line
        self.marker, = ax.plot([0], [0], marker="o", color="crimson", zorder=3)
        self.txt = ax.text(0.7, 0.9, '')
        self.plot_costs()

    def mouse_move(self, event):
        if not event.inaxes is self.ax: return
        # We round xdata for UX
        x, y = event.xdata, event.ydata
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
        self.x_cursor = x

    def plot_costs(self):
        if self.list_labels is None:
            self.list_labels = ["window %s" % i for i in np.arange(1, len(list_costs) + 1)]

        min_y = max_y = np.nan
        for cost, label in zip(self.list_cost_volume, self.list_labels):
            self.ax.plot(self.x_data, cost[self.x_ref, self.y_ref, :], label=label, linestyle="-.")
            min_y = np.nanmin(np.hstack([min_y, cost[self.x_ref, self.y_ref, :]]))
            max_y = np.nanmax(np.hstack([max_y, cost[self.x_ref, self.y_ref, :]]))
        margin = 0.05 * (max_y - min_y)
        self.ax.set_ylim(min_y - margin, max_y + margin)
        self.ax.legend()
        self.ax.set_title("Cost curve")
        self.ax.grid(True, axis='y')

        x_tick_coordinates = np.array(self.ax.get_xticks())
        self.ax.set_xticks(ticks=x_tick_coordinates, labels=(x_tick_coordinates + self.disp_range[0]).astype(int))
        self.ax.set_xlim(self.x_data[0]-self.padd[0], self.x_data[-1]+self.padd[0])


class FullImage(object):
    def __init__(self, ax, x, y, img, padd=[0, 0], title=""):
        """
        An interactive object to plot a full left image and a small red window on the pixel considered.
        Clicking on the window should update the pixel considered
        Args:
            ax: the pyplot Axis on witch the image appears
            x: the x coordinate of the pixel in the image (rows). CAREFUL: it has nothing to do with x_data !
            y: the y coordinate of the pixel in the image (column). CAREFUL: it should correspond to X_ref
            img: the left image with padding (cv2 greyscale image)
            padd: the padding of the image
            title: the title of the subplot
        """
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

        self.reload_figure = False

    def create_rectangle(self):
        return patches.Rectangle((self.y, self.x), self.padd[0], self.padd[1], color="r")

    def update_xy_ref(self, event):
        if event.inaxes is not self.ax: return
        y, x = event.xdata, event.ydata

        # Changing the x and y coordinates but not allowing them to be in padding areas
        x = np.min([self.img.shape[1] - 1 - self.padd[1], x])
        self.x = np.max([self.padd[1], x])
        y = np.min([self.img.shape[0] - 1 - self.padd[0], y])
        self.y = np.max([self.padd[0], y])

        self.reload_figure = True
        plt.close()


class ImageIcon(object):
    def __init__(self, ax, x, y, curs, img, disp_range, padd=[0, 0], title="", connect=True):
        """
        Interactive plot showing a small window around the pixel considered
        Args:
            ax: the pyplot Axis on witch the image appears
            x: the x coordinate of the pixel in the image (rows). CAREFUL: it has nothing to do with x_data !
            y: the y coordinate of the pixel in the image (column). CAREFUL: it should correspond to X_ref
            curs: the cursor object to update X_ref
            img: the left image with padding (cv2 greyscale image)
            padd: the padding of the image
            title: the title of the subplot
        """
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
        y = self.y + self.disp[0] + self.cursor.x_cursor
        # The processed image already has some padding, so we need to take that into account
        self.ax.imshow(self.img[self.x: self.x + 2*self.padd[1] + 1,
                                y: y + 2*self.padd[0] + 1], vmin=0.0, vmax=255.)
        self.ax.figure.canvas.draw_idle()


class ImageBand(object):
    def __init__(self, ax, x, y, curs, img, disp_range, padd=[0, 0], title=""):
        """
        Interactive plot showing all the windows in disparity range considered for the cost volume computation
        Args:
            ax: the pyplot Axis on witch the image appears
            x: the x coordinate of the pixel in the image (rows). CAREFUL: it has nothing to do with x_data !
            y: the y coordinate of the pixel in the image (column). CAREFUL: it should correspond to X_ref
            curs: the cursor object to update X_ref
            img: the left image with padding (cv2 greyscale image)
            padd: the padding of the image
            title: the title of the subplot
        """
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
        self.ly.set_xdata(self.cursor.x_cursor - 0.5)
        self.ry.set_xdata(self.cursor.x_cursor + 2*self.padd[0] + 0.5)
        self.ax.figure.canvas.draw_idle()


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

    disparity_range = [-60, 0]
    padding = [2, 2]

    left_image = prepare_image(left_image_path, padding)
    right_image = prepare_image(right_image_path, padding)

    Y_ref = left_image.shape[0]//2
    X_ref = left_image.shape[1]//2

    continue_loop = True
    while continue_loop:
        fig = plt.figure(figsize=[16., 7.])
        fig.tight_layout()

        ax_cost = plt.subplot(411)
        ax_band = plt.subplot(412)
        ax_full_image = plt.subplot(223)
        ax_left = plt.subplot(247)
        ax_right = plt.subplot(248)


        # Adding the full Left image
        cursor = CostCurveAndCursor(ax_cost, list_costs, X_ref, Y_ref, disparity_range, padd=padding)
        # Adding left and right small patches images
        left_image_patch = ImageIcon(ax_left, X_ref, Y_ref, cursor, left_image,
                                     disparity_range, padd=padding, title="Left patch", connect=False)
        right_image_patch = ImageIcon(ax_right, X_ref, Y_ref, cursor, right_image,
                                      disparity_range, padd=padding, title="Right patch")

        # Adding the right image band
        image_band = ImageBand(ax_band, X_ref, Y_ref, cursor, right_image,
                               disparity_range, padd=padding, title="Right Image on disparity interval")
        full_image = FullImage(ax_full_image, X_ref, Y_ref, left_image, padd=padding, title="Left Image")

        # Connecting pyplot events to figures
        cid_curve = fig.canvas.mpl_connect('motion_notify_event', cursor.mouse_move)
        cid_patch = fig.canvas.mpl_connect('motion_notify_event', right_image_patch.mouse_move)
        cid_band = fig.canvas.mpl_connect('motion_notify_event', image_band.mouse_move)
        cid_full_image = fig.canvas.mpl_connect('button_press_event', full_image.update_xy_ref)

        plt.show()

        X_ref, Y_ref = round(full_image.x) - full_image.padd[1], round(full_image.y) - full_image.padd[0]
        continue_loop = full_image.reload_figure



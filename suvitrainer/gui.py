import tkinter as tk
from tkinter import *  # bad idea... but I did it
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import path
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.widgets import LassoSelector
from skimage import draw
from sunpy import sun, time

from suvitrainer.fileio import Outgest
from suvitrainer.config import Config

matplotlib.use("TkAgg")


class VerticalScrolledFrame(Frame):
    """
    A pure Tkinter scrollable frame that actually works!
    * Use the 'interior' attribute to place widgets inside the scrollable frame
    * Construct and pack/place/grid normally
    * This frame only allows vertical scrolling
    """

    def __init__(self, parent, *args, **kw):
        Frame.__init__(self, parent, *args, **kw)

        # create a canvas object and a vertical scrollbar for scrolling it
        vscrollbar = Scrollbar(self, orient=VERTICAL)
        vscrollbar.pack(fill=Y, side=RIGHT, expand=FALSE)
        canvas = Canvas(self, bd=0, highlightthickness=0,
                        yscrollcommand=vscrollbar.set)
        canvas.pack(side=LEFT, fill=BOTH, expand=TRUE)
        vscrollbar.config(command=canvas.yview)

        # reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # create a frame inside the canvas which will be scrolled with it
        self.interior = interior = Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior,
                                           anchor=NW)

        # track changes to the canvas and frame width and sync them,
        # also updating the scrollbar
        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                canvas.config(width=interior.winfo_reqwidth())

        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())

        canvas.bind('<Configure>', _configure_canvas)


class CustomToolbar(NavigationToolbar2TkAgg):
    """
    A custom Matplotlib toolbar that allows for a lasso selection when no tool is selected
    """
    def __init__(self, canvas_, parent_, edit_frame, app):
        self.parent_frame = parent_
        self.edit_frame = edit_frame
        self.toolitems = (
            ('Home', "Reset zoom", 'home', 'home'),
            ('Back', 'Undo one zoom step', 'back', 'back'),
            ('Forward', 'Redo one zoom step', 'forward', 'forward'),
            (None, None, None, None),
            (None, None, None, None),
            (None, None, None, None),
            ('Pan', 'Activate pan', 'move', 'pan'),
            ('Zoom', 'Activate zoom', 'zoom_to_rect', 'zoom'),
            # ("Lasso", "Activate lasso", "hand", "lasso")
        )
        self.app = app
        NavigationToolbar2TkAgg.__init__(self, canvas_, parent_)

    def pan(self):
        NavigationToolbar2TkAgg.pan(self)
        if self._active:
            self.edit_frame.config(background='white', text='Pan')
        else:
            self.edit_frame.config(background='red', text='Draw')
            self.app.change_class()

    def zoom(self):
        NavigationToolbar2TkAgg.zoom(self)
        if self._active:
            self.edit_frame.config(background='white', text='Zoom')
        else:
            self.edit_frame.config(background='red', text='Draw')
            self.app.change_class()


class App(tk.Tk):
    def __init__(self, data, output, group, image_directory, headers, config_path,
                 blank=False, relabel=None, resizable=True):
        self.config = Config(config_path)
        self.history = []
        self.header = headers[self.config['default']['header']]
        self.headers = headers
        self.interpret_header()
        self.group = group
        self.image_directory = image_directory
        self.blank = blank
        self.relabel = relabel

        tk.Tk.__init__(self)
        self.assign_defaults()  # setup the graphical defaults as desired
        self.output = output  # where to save the trained fits image
        self.shape = (1280, 1280)
        self.data = data
        self.title("The STU: SUVI Training Utility")
        # self.geometry("800x900+500+200")
        self.geometry("800x780")
        # self.make_topmost()
        self.protocol("WM_DELETE_WINDOW", self.on_exit)
        self.resizable(resizable, resizable)
        self.make_gui()

        if blank:
            self.selection_array[:, :] = self.config.solar_class_index['unlabeled']
        else:  # make default
            self.draw_default()
            self.fig.canvas.draw()

        if self.relabel:  # use a previously drawn map
            self.selection_array = self.relabel

    def assign_defaults(self):
        '''assigns variables for many graphical defaults'''
        self.single_color_theme = 'yellow'  # color used for the single color menus
        self.canvas_size = (10, 5)  # size of the canvas frame (in inches)
        self.subplot_grid_spec = {'wspace': 0, 'hspace': 0, 'left': 0, 'bottom': 0, 'right': 1,
                                  'top': 1}  # spacing of subplots in canvas

    def interpret_header(self):

        # handle special cases since date-obs field changed names
        if 'DATE_OBS' in self.header:
            self.date = self.header['DATE_OBS']
        elif 'DATE-OBS' in self.header:
            self.date = self.header['DATE-OBS']
        else:
            raise Exception("Image does not have a DATE_OBS or DATE-OBS field" + str(list(self.group.keys())))

        self.cy, self.cx = self.header['CRPIX1'], self.header['CRPIX2']
        sun_radius_angular = sun.solar_semidiameter_angular_size(t=time.parse_time(self.date)).arcsec
        arcsec_per_pixel = self.header['CDELT1']
        self.sun_radius_pixel = (sun_radius_angular / arcsec_per_pixel)

    def save(self):
        Outgest(self.output, self.selection_array.astype('uint8'), self.headers).save()

    def on_exit(self):
        """When you click to exit, this function is called"""
        answer = messagebox.askyesnocancel("Exit", "Do you want to save as you quit the application?")
        if answer:
            self.save()
            self.quit()
            self.destroy()
        elif answer is None:
            pass
        else:
            self.quit()
            self.destroy()

    def make_topmost(self):
        """Makes this window the topmost window"""
        self.lift()
        self.attributes("-topmost", 1)
        # self.attributes("-topmost", 0)

    def make_gui(self):
        '''Setups the general structure of the gui, the first function called'''
        self.option_window = Toplevel()
        self.option_window.protocol("WM_DELETE_WINDOW", self.on_exit)
        self.canvas_frame = tk.Frame(self, height=500)
        self.option_frame = tk.Frame(self.option_window, height=300)  # self.option_frame_height)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)  # True)
        self.option_frame.pack(side=tk.RIGHT, fill=None, expand=False)
        self.make_options_frame()
        self.make_canvas_frame()
        self.disable_singlecolor()

    def configure_threecolor_image(self, scale=False):
        ''' configures the three color image according to the requested parameters '''
        order = {'red': 0, 'green': 1, 'blue': 2}
        self.image = np.zeros((self.shape[0], self.shape[1], 3))
        for color, var in self.multicolorvars.items():
            channel = var.get()
            self.image[:, :, order[color]] = self.data[channel]
            self.image[:, :, order[color]] = np.power(self.image[:, :, order[color]],
                                                      self.multicolorpower[color].get())
            if scale:
                lower_limit = np.nanpercentile(self.image[:, :, order[color]], 1)
                upper_limit = np.nanpercentile(self.image[:, :, order[color]], 99)
                self.image[np.where(self.image[:, :, order[color]] < lower_limit)] = lower_limit
                self.image[np.where(self.image[:, :, order[color]] > upper_limit)] = upper_limit

        for color, index in order.items():
            self.image[:, :, index] /= np.nanmax(self.image[:, :, index])

    def configure_singlecolor_image(self, scale=False):
        ''' configures the three color image according to the requested parameters '''
        self.image = self.data[self.singlecolorvar.get()]
        self.image = np.power(self.image, self.singlecolorpower.get())

        if scale:
            lower_limit = np.nanpercentile(self.image, 1)
            upper_limit = np.nanpercentile(self.image, 99)
            self.image[np.where(self.image < lower_limit)] = lower_limit
            self.image[np.where(self.image > upper_limit)] = upper_limit

        self.image /= np.nanmax(self.image)

    def updateArray(self, array, indices, value):
        ''' Updates array so that pixels at indices take on the value '''
        lin = np.arange(array.size)
        newArray = array.flatten()
        newArray[lin[indices]] = value
        return newArray.reshape(array.shape)

    def onlasso(self, verts):
        ''' Main function to control the action of the lasso '''
        global selection_array, pix, solar_class_var  # TODO is global really needed here?
        p = path.Path(verts)
        ind = p.contains_points(self.pix, radius=1)
        self.history.append(self.selection_array.copy())
        self.selection_array = self.updateArray(self.selection_array,
                                                ind,
                                                self.solar_class_var.get())
        self.mask.set_data(self.selection_array)
        self.fig.canvas.draw_idle()

    def make_canvas_frame(self):
        ''' Create the canvas frame for the first time '''

        self.fig, (self.imageax, self.previewax) = plt.subplots(ncols=2,
                                                                figsize=self.canvas_size,
                                                                sharex=True, sharey=True,
                                                                gridspec_kw=self.subplot_grid_spec)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.imageax.plot()  # TODO is this really needed?
        self.configure_threecolor_image()
        self.imageplot = self.imageax.imshow(self.image)
        self.imageax.set_xlim([0, self.shape[0]])
        self.imageax.set_ylim([0, self.shape[0]])
        self.imageax.set_aspect("equal")  # TODO is this really needed?
        self.imageax.set_axis_off()

        self.selection_array = np.zeros(self.shape, dtype=np.uint8)
        if self.blank:
            self.selection_array += self.config.solar_class_index['unlabeled']
        else:
            self.draw_default()
        self.history.append(self.selection_array)
        # ordered_solar_classes = dict([(index, label) for label, index in SOLAR_CLASSES])
        colortable = [self.config.solar_colors[self.config.solar_class_name[i]]
                      for i in range(len(self.config.solar_classes))]
        cmap = matplotlib.colors.ListedColormap(colortable)
        self.mask = self.previewax.imshow(self.selection_array,
                                          origin='lower',
                                          interpolation='nearest',
                                          cmap=cmap,
                                          vmin=-1, vmax=len(colortable))
        self.previewax.set_xlim([0, self.shape[0]])
        self.previewax.set_ylim([0, self.shape[0]])
        self.previewax.set_aspect("equal")
        self.previewax.set_axis_off()

        self.pix = np.arange(self.shape[0])  # assumes square image
        xv, yv = np.meshgrid(self.pix, self.pix)
        self.pix = np.vstack((xv.flatten(), yv.flatten())).T

        lineprops = dict(color="red", linewidth=2)
        self.lasso = LassoSelector(self.imageax, self.onlasso, lineprops=lineprops)

        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbarcenterframe = tk.LabelFrame(self.canvas_frame,
                                                borderwidth=0,
                                                text="Draw: unlabeled",
                                                relief=tk.FLAT,
                                                labelanchor=tk.N,
                                                background='red')
        toolbarframe = tk.Frame(self.toolbarcenterframe)
        toolbar = CustomToolbar(self.canvas, toolbarframe, self.toolbarcenterframe, self)

        toolbar.update()
        self.fig.canvas.toolbar.set_message = lambda x: ""  # remove state reporting
        toolbarframe.pack()
        self.toolbarcenterframe.pack(side=tk.BOTTOM, fill=tk.X)

    def make_options_frame(self):
        self.tab_frame = ttk.Notebook(self.option_frame, width=800)
        self.tab_configure = tk.Frame(self.tab_frame)
        self.tab_classify = tk.Frame(self.tab_frame)
        self.make_configure_tab()
        self.make_classify_tab()

        self.tab_frame.add(self.tab_configure, text="Configure")
        self.tab_frame.add(self.tab_classify, text="Classify")
        self.tab_frame.pack(fill=tk.BOTH, expand=True)

    def disable_multicolor(self):
        ''' swap from the multicolor image to the single color image '''
        # disable the multicolor image
        for color in ['red', 'green', 'blue']:
            self.multicolorscales[color].config(state=tk.DISABLED, bg='grey')
            self.multicolorframes[color].config(bg='grey')
            self.multicolorlabels[color].config(bg='grey')
            self.multicolordropdowns[color].config(bg='grey', state=tk.DISABLED)
            self.multicolorminscale[color].config(bg='grey', state=tk.DISABLED)
            self.multicolormaxscale[color].config(bg='grey', state=tk.DISABLED)

        # enable the single color
        self.singlecolorscale.config(state=tk.NORMAL, bg=self.single_color_theme)
        self.singlecolorframe.config(bg=self.single_color_theme)
        self.singlecolorlabel.config(bg=self.single_color_theme)
        self.singlecolordropdown.config(bg=self.single_color_theme, state=tk.NORMAL)
        self.singlecolorminscale.config(bg=self.single_color_theme, state=tk.NORMAL)
        self.singlecolormaxscale.config(bg=self.single_color_theme, state=tk.NORMAL)

    def disable_singlecolor(self):
        ''' swap from the single color image to the multicolor image '''
        # enable the multicolor
        for color in ['red', 'green', 'blue']:
            self.multicolorscales[color].config(state=tk.NORMAL, bg=color)
            self.multicolorframes[color].config(bg=color)
            self.multicolorlabels[color].config(bg=color)
            self.multicolordropdowns[color].config(bg=color, state=tk.NORMAL)
            self.multicolorminscale[color].config(bg=color, state=tk.NORMAL)
            self.multicolormaxscale[color].config(bg=color, state=tk.NORMAL)

        # disable the singlecolor
        self.singlecolorscale.config(state=tk.DISABLED, bg='grey')
        self.singlecolorframe.config(bg='grey')
        self.singlecolorlabel.config(bg='grey')
        self.singlecolordropdown.config(bg='grey', state=tk.DISABLED)
        self.singlecolorminscale.config(bg="grey", state=tk.DISABLED)
        self.singlecolormaxscale.config(bg="grey", state=tk.DISABLED)

    def update_button_action(self):
        if self.mode.get() == 3:  # threecolor
            self.configure_threecolor_image()
        elif self.mode.get() == 1:  # singlecolor
            self.configure_singlecolor_image()
        else:
            raise ValueError("mode can only be singlecolor or threecolor")

        self.imageplot.set_data(self.image)
        if self.mode.get() == 1:  # singlecolor
            self.imageplot.set_cmap('gist_gray')
        self.fig.canvas.draw_idle()

    def make_configure_tab(self):
        # Setup the choice between single and multicolor
        modeframe = tk.Frame(self.tab_configure)
        self.mode = tk.IntVar()
        singlecolor = tk.Radiobutton(modeframe, text="Single color", variable=self.mode,
                                     value=1, command=lambda: self.disable_multicolor())
        multicolor = tk.Radiobutton(modeframe, text="Three color", variable=self.mode,
                                    value=3, command=lambda: self.disable_singlecolor())
        self.mode.set(3)
        singlecolor.pack(side=tk.LEFT)
        multicolor.pack(side=tk.LEFT)

        updatebutton = tk.Button(master=modeframe, text="Update",
                                 command=self.update_button_action)
        updatebutton.pack(side=tk.RIGHT)
        modeframe.grid(row=0, column=0)
        self.setup_multicolor()
        self.setup_singlecolor()

    def setup_singlecolor(self):
        self.singlecolorframe = tk.Frame(self.tab_configure, bg=self.single_color_theme)
        channel_choices = sorted(list(self.data.keys()))
        self.singlecolorlabel = tk.Label(self.singlecolorframe, text="single", bg=self.single_color_theme, width=10)
        self.singlecolorvar = tk.StringVar()
        self.singlecolorpower = tk.DoubleVar()
        self.singlecolormin = tk.DoubleVar()
        self.singlecolormax = tk.DoubleVar()
        self.singlecolordropdown = tk.OptionMenu(self.singlecolorframe, self.singlecolorvar, *channel_choices)
        self.singlecolorscale = tk.Scale(self.singlecolorframe, variable=self.singlecolorpower,
                                         orient=tk.HORIZONTAL, from_=self.config.ranges['single_color_power_minx'],
                                         bg=self.single_color_theme,
                                         to_=self.config.ranges['single_color_power_max'],
                                         resolution=self.config.ranges['single_color_power_resolution'],
                                         length=200)
        # self.singlecolorminscale = tk.Scale(self.singlecolorframe, variable=self.singlecolormin,
        #                                     orient=tk.HORIZONTAL, from_=SINGLECOLOR_VRANGE[0],
        #                                     bg=self.single_color_theme,
        #                                     to_=SINGLECOLOR_VRANGE[1], resolution=SINGLECOLOR_VRESOLUTION, length=200)
        #
        # self.singlecolormaxscale = tk.Scale(self.singlecolorframe, variable=self.singlecolormax,
        #                                     orient=tk.HORIZONTAL, from_=SINGLECOLOR_VRANGE[0],
        #                                     bg=self.single_color_theme,
        #                                     to_=SINGLECOLOR_VRANGE[1], resolution=SINGLECOLOR_VRESOLUTION, length=200)

        self.singlecolorvar.set(self.config.default['single'])
        self.singlecolorpower.set(self.config.default['single_power'])
        #self.singlecolormin.set(DEFAULT_VMIN['single'])
        #self.singlecolormax.set(DEFAULT_VMAX['single'])
        self.singlecolordropdown.config(bg=self.single_color_theme, width=10)
        self.singlecolorlabel.pack(side=tk.LEFT)
        self.singlecolorscale.pack(side=tk.RIGHT)
        # self.singlecolormaxscale.pack(side=tk.RIGHT)
        # self.singlecolorminscale.pack(side=tk.RIGHT)
        self.singlecolordropdown.pack()
        self.singlecolorframe.grid(row=4, columnspan=5, rowspan=1)

    def setup_multicolor(self):
        # Setup the options for multicolor
        multicolormasterframe = tk.Frame(self.tab_configure)
        channel_choices = sorted(list(self.data.keys()))
        rgb = ['red', 'green', 'blue']
        self.multicolorframes = {color: tk.Frame(multicolormasterframe, bg=color) for color in rgb}
        self.multicolorlabels = {color: tk.Label(self.multicolorframes[color], text=color, bg=color, width=10) for color
                                 in rgb}
        self.multicolorvars = {color: tk.StringVar() for color in rgb}
        self.multicolorpower = {color: tk.DoubleVar() for color in rgb}
        self.multicolormin = {color: tk.DoubleVar() for color in rgb}
        self.multicolormax = {color: tk.DoubleVar() for color in rgb}

        self.multicolordropdowns = {color: tk.OptionMenu(self.multicolorframes[color],
                                                         self.multicolorvars[color],
                                                         *channel_choices) for color in rgb}

        self.multicolorscales = {color: tk.Scale(self.multicolorframes[color],
                                                 variable=self.multicolorpower[color],
                                                 orient=tk.HORIZONTAL,
                                                 from_=self.config.ranges['multi_color_power_min'],
                                                 to_=self.config.ranges['multi_color_power_max'], bg=color,
                                                 resolution=self.config.ranges['multi_color_power_resolution'],
                                                 length=200) for color in rgb}
        # self.multicolorminscale = {color: tk.Scale(self.multicolorframes[color],
        #                                            variable=self.multicolormin[color],
        #                                            orient=tk.HORIZONTAL, from_=MULTICOLOR_VRANGE[0],
        #                                            to_=MULTICOLOR_VRANGE[1], bg=color,
        #                                            resolution=MULTICOLOR_VRESOLUTION, length=200) for color in rgb}
        # self.multicolormaxscale = {color: tk.Scale(self.multicolorframes[color],
        #                                            variable=self.multicolormax[color],
        #                                            orient=tk.HORIZONTAL, from_=MULTICOLOR_VRANGE[0],
        #                                            to_=MULTICOLOR_VRANGE[1], bg=color,
        #                                            resolution=MULTICOLOR_VRESOLUTION, length=200) for color in rgb}

        for color in rgb:
            self.multicolorvars[color].set(self.config.default[color])
            self.multicolorpower[color].set(self.config.default[color + "_power"])
            # self.multicolormin[color].set(DEFAULT_VMIN[color])
            # self.multicolormax[color].set(DEFAULT_VMAX[color])
            self.multicolordropdowns[color].config(bg=color, width=10)
            self.multicolorlabels[color].pack(side=tk.LEFT)

            self.multicolorscales[color].pack(side=tk.RIGHT)
            # self.multicolormaxscale[color].pack(side=tk.RIGHT)
            # self.multicolorminscale[color].pack(side=tk.RIGHT)
            self.multicolordropdowns[color].pack()
            self.multicolorframes[color].pack(fill=tk.BOTH)
        multicolormasterframe.grid(row=1, column=0, columnspan=5, rowspan=3)

    def undobutton_action(self):
        if len(self.history) > 1:
            old = self.history.pop(-1)
            self.selection_array = old
            self.mask.set_data(old)
            self.fig.canvas.draw_idle()

    def change_class(self):
        ''' on changing the classification label, update the "draw" text '''
        self.toolbarcenterframe.config(text="Draw: {}".format(self.config.solar_class_name[self.solar_class_var.get()]))

    def make_classify_tab(self):
        self.pick_frame = tk.Frame(self.tab_classify)
        self.pick_frame2 = tk.Frame(self.tab_classify)
        # self.pick_frame = VerticalScrolledFrame(self.tab_classify) #
        # self.pick_frame2 = VerticalScrolledFrame(self.tab_classify) #
        # self.pick_label = tk.Label(self.pick_frame,
        #                           text="Choose a label:",
        #                           font=BOLDFONT).pack()
        self.solar_class_var = tk.IntVar()
        self.solar_class_var.set(0)  # initialize to unlabeled
        buttonnum = 0
        frame = [self.pick_frame, self.pick_frame2]
        for text, value in self.config.solar_classes:
            b = tk.Radiobutton(frame[buttonnum % 2], text=text,
                               # b = tk.Radiobutton(frame[buttonnum%2].interior, text=text,
                               variable=self.solar_class_var,
                               value=value, background=self.config.solar_colors[text],
                               indicatoron=0, width=50, height=2, command=self.change_class)
            b.pack(fill=tk.BOTH, expand=1)
            buttonnum += 1
            #
            # if buttonnum %2 == 0:
            #    b.pack(anchor=tk.W)
            # else:
            #    b.pack(anchor=tk.E)
            # buttonnum += 1

        self.pick_frame.grid(row=0, column=0, rowspan=5, sticky=tk.W + tk.E + tk.N + tk.S)
        self.pick_frame2.grid(row=0, column=1, rowspan=5, sticky=tk.W + tk.E + tk.N + tk.S)
        # self.pick_frame.pack(side=tk.LEFT, anchor=tk.NW, fill=tk.X, expand=1)#grid(row=0, column=0)
        # self.pick_frame2.pack(side=tk.RIGHT, anchor=tk.NE, fill=tk.X, expand=1)
        undobutton = tk.Button(master=self.tab_classify, text="Undo",
                               command=self.undobutton_action)
        undobutton.grid(row=6, column=0, columnspan=2, sticky=tk.W + tk.E)
        # undobutton.pack(side=tk.TOP, anchor=tk.N, fill=tk.X, expand=1)

        # self.confidence_frame = tk.Frame(self.tab_classify)
        # self.confidence_label = tk.Label(self.confidence_frame,
        #                                  text="Choose a confidence level:",
        #                                  font=BOLDFONT).pack()
        # self.confidence_var = tk.IntVar()
        # self.confidence_var.set(3) # initialized to midpoint
        # for text, value in CONFIDENCE_LEVELS:
        #     b = tk.Radiobutton(self.confidence_frame, text=text,
        #                        variable=self.confidence_var, value=value)
        #     b.pack(anchor=tk.W)
        # self.confidence_frame.pack(side=tk.RIGHT)#grid(row=0, column=1)

    def draw_circle(self, center, radius, array, value, mode="set"):
        ri, ci = draw.circle(center[0], center[1],
                             radius=radius,
                             shape=array.shape)

        existing = array[ri, ci]
        if mode == "add":
            array[ri, ci] += value
        elif mode == "set":
            array[ri, ci] = value
        return ri, ci, existing

    def draw_annulus(self, center, inner_radius, outer_radius, array, value, mode="set"):
        if mode == "add":
            self.draw_circle(center, outer_radius, array, value)
            self.draw_circle(center, inner_radius, array, -value)
        elif mode == "set":
            ri, ci, existing = self.draw_circle(center, inner_radius, array, -value)
            self.draw_circle(center, outer_radius, array, value)
            array[ri, ci] = existing

    def draw_default(self):
        ''' Draw suggested sun disk, limb, and empty background '''
        self.selection_array[:, :] = self.config.solar_class_index['empty_outer_space']
        self.draw_annulus((self.cx, self.cy),
                          self.sun_radius_pixel - 5,
                          self.sun_radius_pixel + 15,
                          self.selection_array,
                          self.config.solar_class_index['limb'])
        self.draw_circle((self.cx, self.cy),
                         self.sun_radius_pixel - 5,
                         self.selection_array,
                         self.config.solar_class_index['quiet_sun'])



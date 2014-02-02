# Import plotting functions
import matplotlib.pyplot as plt
# Background pyxflow module
from . import _pyxflow as px

# Line collection
from matplotlib.collections import LineCollection
# Color conversion
from matplotlib.colors import colorConverter
# Import the customizable colormap class
from matplotlib.colors import LinearSegmentedColormap

# Class for xf_Plot objects
class xf_Plot:

    # Class initialization file.
    def __init__(self, All=None, fname=None, **kwargs):
        """
        h_p = xf_Plot(All=None, fname=None, **kwargs)

        INPUTS:
           All   : xf_All object
           fname : path to a '.xfa' file

        OUTPUTS:
           h_p   : instance of xf_Plot class

        KEYWORD ARGUMENTS:
           (See xf_Plot.Plot)
        """

        # Load the xf_All object if not given directly.
        if All is None:
            All = xf_All(fname)

        # Initialize some handles.
        self.figure = None
        self.axes = None
        self.Scalar = None
        self.Mesh = None

        # Produce the initial plot.
        #self.Plot(All, **kwargs)
        self.PlotMesh(All.Mesh, **kwargs)

    # Method to draw the scalar plot
    def Plot(self, All, xyrange=None, vgroup='State', scalar=None, **kwargs):
        """
        h_p = xf_Plot.Plot(All, xyrange=None, vgroup='State', scalar=None, **kwargs)

        INPUTS:
           All     : xf_All object
           xyrange : list of coordinates to plot (Note 1)
           vgroup  : title of vector group to use
           scalar  : name of scalar to plot (Note 2)

        OUTPUTS:
           h_t : <matplotlib.pyplot.tripcolor> instance

        KEYWORD ARGUMENTS:
           mesh      : flag to draw a mesh
           AutoScale : automatically resize figure to fit data range


        This is the plotting method for the xf_All class.  More capabilities
        will be added.

        NOTES:
           (1) The 'xyrange' keyword is specified in the form

                   [xmin, xmax, ymin, ymax]

               However, inputs such as `xyrange=(0,0,None,None)` are also
               acceptable.  In this case, the minimum value for both
               coordinates will be zero, but no maximum value will be
               specified.  Furthermore, alternate keys 'xmin', 'xmax', etc.
               override the values specified in 'range'.

           (2) The 'scalar' keyword may be any state that's in the cell
               interior list of states.  If the equation set is Navier-Stokes,
               the options Mach number, entropy, and pressure are also
               available.
        """
        # Versions:
        #  2013-09-29 @dalle   : First version

        # Check for a DataSet
        if not (All.DataSet.nData >= 1):
            raise IndexError("No DataSet found.")
        # Check that we have a vector group.
        if not (All.DataSet.Data[0].Type == 'VectorGroup'):
            raise TypeError("DataSet is not a xf_VectorGroup instance.")
        # This is for 2D right now!
        if All.Mesh.Dim != 2:
            raise NotImplementedError("3D plotting is not implemented.")

        # Process dimension kwargs.
        xmin = kwargs.get('xmin')
        xmax = kwargs.get('xmax')
        ymin = kwargs.get('ymin')
        ymax = kwargs.get('ymax')

        # Process the window for plotting.
        if xyrange is not None:
            # Don't override values directly specified.
            if xmin is None:
                xmin = xyrange[0]
            if xmax is None:
                xmax = xyrange[1]
            if ymin is None:
                ymin = xyrange[2]
            if ymax is None:
                ymax = xyrange[3]
        # Make sure the values are not None before handing to C function.
        if xmin is None:
            xmin = All.Mesh.Coord[:, 0].min()
        if xmax is None:
            xmax = All.Mesh.Coord[:, 0].max()
        if ymin is None:
            ymin = All.Mesh.Coord[:, 1].min()
        if ymax is None:
            ymax = All.Mesh.Coord[:, 1].max()

        # Get the titles of the vector groups available.
        UG_Titles = [D.Title for D in All.DataSet.Data]
        # Check for the requested vector group.
        if vgroup in UG_Titles:
            # Get the matching vector group.
            UG = All.DataSet.Data[UG_Titles.index(vgroup)].Data
        elif vgroup is None:
            # Get the first vector group.
            UG = All.DataSet.Data[0].Data
        else:
            # Error
            raise RuntimeError(
                "Unrecognized DataSet title '{}'".format(vgroup))

        # Limits on plot window
        xlim = [xmin, xmax, ymin, ymax]
        # Save the limits.
        self.xlim = xlim
        Order = kwargs.get('order')

        x, y, c = px.MeshPlotData(self._ptr, xmin, xmax, Order)
        
        # Get the calculated vector, triangulation, and mesh lines.
        X, u, T, L = px.PlotData(All._ptr, UG._ptr, xlim)
        
        # Convert mesh lines to NumPy array.
        L = np.asarray(L)

        # Pull the first vector.
        U = UG.Vector[0]
        # Get the scalar.
        M = U.get_scalar(u, scalar)

        # Draw the requested scalar.
        h_t = plt.tripcolor(X[:, 0], X[:, 1], T, M, shading='gouraud')

        # Save the figure and axis handles.
        if kwargs.get('figure') is not None:
            self.figure = kwargs['figure']
        else:
            self.figure = plt.figure()

        if kwargs.get('axes') is not None:
            self.axes = kwargs['axes']
        else:
            self.axes = self.figure.gca()

        # Check for a grid.
        if kwargs.get('mesh', True) is True:
            # Nx2 matrix of xy-coordinates for each element
            xx = (X[j, :] for j in L)
            # Make a collection of lines with the same properties.
            h_l = LineCollection(xx, linewidths=0.2, colors=(0, 0, 0, 1))
            # Add all the lines at once.
            plt.gca().add_collection(h_l)
            # Save the handle.
            self.mesh = h_l
        else:
            # Save an empty handle.
            self.mesh = None

        # Check whether or not to fill the window
        if kwargs.get('fill', False) is True:
            self.FillWindow()

        # Autoscale the figure window.
        if kwargs.get('AutoScale', True) is True:
            self.AutoScale()

        # return the handle
        self.state = h_t
        
    
    # Plot method for mesh
    def PlotMesh(self, Mesh, **kwargs):
        """
        Mesh plotting method
        """
        
        # Initialize coordinate limits.
        xLimMin = [None for i in range(Mesh.Dim)]
        xLimMax = [None for i in range(Mesh.Dim)]
        # Lowest priority: list of xmins
        if kwargs.get('xmin') is not None:
            xmin = kwargs['xmin']
            # Check the dimensions.
            if len(xmin) == Mesh.Dim:
                xLimMin = xmin
        # Lowest priority: list of xmins
        if kwargs.get('xmax') is not None:
            xmax = kwargs['xmax']
            # Check the dimensions.
            if len(xmax) == Mesh.Dim:
                xLimMax = xmax
                
        # Next priority: full list
        if kwargs.get('xlim') is not None:
            xlim = kwargs['xlim']
            # Check the dimensions.
            if len(xlim) == 2*Mesh.Dim:
                # Get every other element.
                xLimMin = xlim[0::2]
                xLimMax = xlim[1::2]
                
        # Second priority, individual limits
        if kwargs.get('xlim') is not None:
            xlim = kwargs['xlim']
            # Check the dimensions.
            if len(xlim)==2 and Mesh.Dim>1:
                xLimMin[0] = xlim[0]
                xLimMax[0] = xlim[1]
        if kwargs.get('ylim') is not None:
            ylim = kwargs['ylim']
            # Check if it's appropriate.
            if len(ylim)==2 and Mesh.Dim>1:
                xLimMin[1] = ylim[0]
                xLimMax[1] = ylim[1]
        if kwargs.get('zlim') is not None:
            zlim = kwargs['zlim']
            # Check if it's appropriate.
            if len(zlim)==2 and Mesh.Dim>2:
                xLimMin[2] = zlim[0]
                xLimMax[2] = zlim[1]
                
        # Top priority: individual mins and maxes overrule other inputs.
        if kwargs.get('xmin') is not None:
            xmin = kwargs['xmin']
            # Check for a scalar (and relevance).
            if len(xmin)==1 and Mesh.Dim>1:
                xLimMin[0] = xmin
        if kwargs.get('xmax') is not None:
            xmax = kwargs['xmax']
            # Check for a scalar (and relevance).
            if len(xmax)==1 and Mesh.Dim>1:
                xLimMax[0] = xmax
        if kwargs.get('ymin') is not None:
            ymin = kwargs['ymin']
            # Check for a scalar.
            if len(ymin)==1:
                xLimMin[1] = ymin
        if kwargs.get('ymax') is not None:
            ymax = kwargs['ymax']
            # Check for a scalar.
            if len(ymax)==1:
                xLimMax[1] = ymax
        if kwargs.get('zmin') is not None:
            zmin = kwargs['zmin']
            # Check for a scalar.
            if len(zmin)==1:
                xLimMin[2] = zmin
        if kwargs.get('zmax') is not None:
            zmax = kwargs['zmax']
            # Check for a scalar.
            if len(zmax)==1:
                xLimMax[2] = zmax

        # Get the defaults based on all mesh coordinates.
        for i in range(Mesh.Dim):
            if xLimMin[i] is None:
                xLimMin[i] = Mesh.Coord[:, i].min()
            if xLimMax[i] is None:
                xLimMax[i] = Mesh.Coord[:, i].max()

        if kwargs.get('figure') is not None:
            self.figure = kwargs['figure']
        else:
            self.figure = plt.figure()

        if kwargs.get('axes') is not None:
            self.axes = kwargs['axes']
        else:
            self.axes = self.figure.gca()

        Order = kwargs.get('order')

        x, y, c = px.MeshPlotData(Mesh._ptr, xLimMin, xLimMax, Order)
        s = []
        for f in range(len(c) - 1):
            s.append(zip(x[c[f]:c[f + 1]], y[c[f]:c[f + 1]]))

        line_options = kwargs.get('line_options', {})

        c = LineCollection(s, colors=(0, 0, 0, 1), **line_options)
        self.axes.add_collection(c)

        if kwargs.get('reset_limits', True):
            self.axes.set_xlim(xLimMin[0], xLimMax[0])
            self.axes.set_ylim(xLimMin[1], xLimMax[1])

        self.Mesh = c
    
    
    

    # Method to make plot fill the window
    def FillWindow(self):
        """
        h_p.FillWindow()

        INPUTS:
           h_p : xf_Plot instance, with following properties defined
              .axes : handle to the axes

        OUTPUTS:
           (None)

        This method simply sets the limits of the axes so that it takes up the
        entire figure window.
        """
        # Versions:
        #  2013-12-15 @dalle   : First version

        # Get the axes handle.
        h_a = self.axes
        # Set the position in relative (scaled) units.
        h_a.set_position([0, 0, 1, 1])
        # Update.
        if plt.isinteractive():
            plt.draw()

        return None

    # Method to draw a NEW figure at the correct size
    def AutoScale(self):
        """
        h_p.AutoScale()

        INPUTS:
           h_p : xf_Plot instance, with following properties defined
              .figure : handle to figure/plot window
              .axes   : handle to the axes
              .xlim   : list of [xmin, xmax, ymin, ymax]

        OUTPUTS:
           (None)

        This method has a similar function to `axis('equal')`, but with the
        additional feature that it resizes the window to have a one-to-one
        aspect ratio with the axes limits defined in the xf_Plot object.

        This differs from the behavior of `axis('equal')`, which changes the
        axis limits to preserve the one-to-one ratio instead of changing the
        figure size.
        """

        # Versions:
        #  2013-12-15 @dalle   : First version

        # Extract the limits.
        xlim = self.xlim
        # Get the aspect ratio.
        AR = (xlim[3] - xlim[2]) / (xlim[1] - xlim[0])

        # Get the width of the figure.
        w = self.figure.get_size_inches()[0]
        # Extract the axes sizing info in two steps.
        p_a = self.axes.get_position()
        # Relative width and height (relative to figure size)
        x = p_a.bounds[2]
        y = p_a.bounds[3]

        # Determine the appropriate figure height
        h = AR * x * w / y
        # Set the limits of the plot window.
        plt.axis(xlim)
        # Set the figure size (and update).
        self.figure.set_size_inches(w, h, forward=True)

        return None

    def Show(self):
        self.figure.show()

    # Method to customize the colormap
    def SetColorMap(self, colorList):
        """
        h_p.set_colormap(colorList)

        """
        # Versions:
        #  2013-12-15 @dalle   : First version

        # Use the method below.
        set_colormap(self.state, colorList)
        return None


def set_colormap(h, colorList):
    """
    set_colormap(h, colorList)

    """
    # Versions:
    #  2013-12-15 @dalle   : Introductory version

    # Determine the nature of the handle.
    if not hasattr(h, 'set_cmap'):
        raise AttributeError(
            "Input to set_colormap must have 'set_cmap' attribute.")

    # Check for named colormap
    if isinstance(colorList, str):
        raise IOError(
            "Named color maps are not implemented.")

    # Get dimensions of the list.
    nColors = len(colorList)

    # Determine if break points are specified.
    if len(colorList[0]) == 2:
        # User-specified break points
        vList = [c[1] for c in colorList]
        # Names of the colors
        cList = [c[0] for c in colorList]
    else:
        # Default break points
        vList = np.linspace(0, 1, nColors)
        # Names of the colors
        cList = list(colorList)

    # Copy the color list to make a left and right break points.
    c1List = cList
    c2List = cList
    # Find repeated colors.
    iRepeat = np.nonzero(np.diff(vList) <= 1e-8)
    # Compress the list of breakpoints.
    vList = np.delete(vList, iRepeat)
    # Process the repeated colors in reverse order.
    for i in iRepeat[::-1]:
        # Delete the second color at the same point from the "left" list.
        c1List = np.delete(c1List, i)
        # Delete the first color at the same point from the "right" list.
        c2List = np.delete(c2List, i + 1)

    # Convert the colors to RGB.
    rgba1List = colorConverter.to_rgba_array(c1List)
    rgba2List = colorConverter.to_rgba_array(c2List)

    # Make the lists for the individual components (red, green, blue).
    rList = [(vList[i], rgba1List[i, 0], rgba2List[i, 0])
             for i in range(vList.size)]
    gList = [(vList[i], rgba1List[i, 1], rgba2List[i, 1])
             for i in range(vList.size)]
    bList = [(vList[i], rgba1List[i, 2], rgba2List[i, 2])
             for i in range(vList.size)]
    aList = [(vList[i], rgba1List[i, 3], rgba2List[i, 3])
             for i in range(vList.size)]
    # Make the required dictionary for the colormap.
    cdict = {'red': rList, 'green': gList, 'blue': bList, 'alpha': aList}
    # Set the colormap.
    cm = LinearSegmentedColormap('custom', cdict)
    # Apply the colormap
    h.set_cmap(cm)
    # Return nothing
    return None

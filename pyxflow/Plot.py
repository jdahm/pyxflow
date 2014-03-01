"""
The *Plot* module contains a special class used to create and control pyXFlow's
plotting capabilities.  The data members 
"""

# Background pyxflow module
from . import _pyxflow as px

# Import plotting functions
import matplotlib.pyplot as plt
# Line collection
from matplotlib.collections import LineCollection
# Color conversion
from matplotlib.colors import colorConverter
# Import the customizable colormap class
from matplotlib.colors import LinearSegmentedColormap

# Import matplotlib
from matplotlib import rcParams
# Move those damn ticks to the outside.
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'

# Variable testing
import numpy as np

# Class for xf_Plot objects
class xf_Plot:
    """
    Create a :class:`pyxflow.Plot.xf_Plot` object.
    
    The initialization method for this class can be used as a plot function of
    its own.  By passing an *xf_All* or *xf_Mesh*, the appropriate plotting
    function of that input is called.
    
    :Call:
        >>> h = xf_Plot(All=None, Mesh=None, **kwargs)
    
    :Parameters:
        *All*: :class:`pyxflow.All.xf_All`
            Instance of XFlow all object containing mesh and solution
        *Mesh*: :class:`pyxflow.Mesh.xf_Mesh`
            Object used for plotting mesh without a solution
    
    :Data members:
        *h.figure*: :class:`matplotlib.figure.Figure`
            Handle to the figure in which the XFlow plot will be made
        *h.axes*: :class:`matplotlib.axes.AxesSubplot`
            Handle to the axes in which the XFlow plot will be made
        *h.mesh*: :class:`matplotlib.collections.LineCollection`
            Handle to the mesh lines, if they are drawn
        *h.scalar*: :class:`matplotlib.collections.TriMesh`
            Handle to the background color plot, if it is drawn
        *h.colorbar*: :class:`matplotlib.colorbar.Colorbar`
            Handle to colorbar
        *h.cax*: :class:`matplotlib.axes.AxesSubplot`
            Handle to axes containing colorbar
        *h.xmin*: :class:`float` list
            List of minimum coordinate in the plot window for each dimension
        *h.xmax*: :class:`float` list
            List of maximum coordinate in the plot window for each dimension
    
    :Kwargs:
        The kwargs are passed to :func:`pyxflow.Mesh.Plot()` or
        :func:`pyxflow.All.Plot()` depending on which inputs are given.
    
        See also kwargs for :func:`pyxflow.Plot.GetXLims`
    """

    # Class initialization file.
    def __init__(self, All=None, Mesh=None, **kwargs):
        """
        Initialization method for :class:`pyxflow.Plot.xf_Plot`
        """
        
        # Initialize some handles.
        self.figure = None
        self.axes = None
        self.mesh = None
        self.scalar = None
        self.contour = None
        self.colorbar = None
        self.cax = None
        # Store the limits.
        self.xmin = None
        self.xmax = None
            
        # Determine if an input allows plots to be made.
        if All is not None:
            # This will plot the first argument regardless of whether it's
            # actually an All or Mesh.
            self = All.Plot(**kwargs)
        elif hasattr(Mesh, 'Coord'):
            # Plot the mesh only.
            self = Mesh.Plot(**kwargs)
    
    # Method to remove the plot (or parts of it).
    def remove(self):
        """
        Remove all of the visible features of a plot handle
        
        :Call:
            >>> Plot.remove()
        
        :Parameters:
            *Plot*: :class:`pyxflow.Plot.xf_Plot`
                Handle to XFlow plot to be removed from the active window
        
        :Returns:
            ``None``
        """
        # Check for a mesh plot.
        if self.mesh is not None:
            # Remove it (if it hasn't been already). 
            try:
                self.mesh.remove()
            except:
                pass
            # Get rid of the now-dead line collection.
            self.mesh = None
        # Check for a scalar plot.
        if self.scalar is not None:
            # Remove it (if it hasn't been already). 
            try:
                self.scalar.remove()
            except:
                pass
            # Get rid of the now-dead TriMesh.
            self.scalar = None
        # Check for a mesh plot.
        if self.contour is not None:
            # Remove it (if it hasn't been already). 
            try:
                self.contour.remove()
            except:
                pass
            # Get rid of the now-dead TriMesh.
            self.contour = None
        # Update the plot if appropriate.
        if plt.isinteractive():
            plt.draw()
        # Not output
        return None
        

    
    # Method to show a damn colorbar (why is this so difficult?)
    def ShowColorbar(self, **kwargs):
        """
        Show a colorbar for an *xf_Plot* instance
        
        The plot's *scalar* property will be used for the colorbar by default.
        This will not be the case if ``Plot.scalar`` is ``None``.
        
        :Call:
            >>> Plot.ShowColorbar()
        
        :Parameters:
            *Plot*: :class:`pyxflow.Plot.xf_Plot`
                Handle to XFlow plot to be removed from the active window
        
        :Returns:
            ``None``
        
        :Kwargs:
            All keyword arguments are passed to :func:`colorbar()`
        """
        # Check for something to use.
        if self.scalar is not None:
            # Add a colorbar (worst syntax ever)
            cb = self.figure.colorbar(self.scalar, **kwargs)
        elif self.contour is not None:
            # Add the colorbar using the contour plot
            cb = self.figure.colorbar(self.contour, **kwargs)
        # Save the colorbar handle (easy part).
        self.colorbar = cb
        # Find all the axes (because colorbar wonderfully doesn't give you the
        # axes handle!
        ax = self.figure.get_axes()
        # There should be two axes.  This will completely break if subplots are
        # going on, but there's no way to fix it due to the wonderful way in
        # which matplotlib's colorbar is programmed.
        if ax[0] == self.axes and len(ax) == 2:
            self.cax = ax[1]
        # Update the plot if appropriate.
        if plt.isinteractive():
            plt.draw()
        # Not output
        return None
            
        
    # Method to make plot fill the window
    def FillWindow(self, xfrac=None):
        """
        Modify an *xf_Plot* instance to fill the entire window
        
        Fills 90% of window if a colorbar is present
        
        :Call:
            >>> Plot.FillWindow(xfrac=1.0)

        :Parameters:
            *Plot*: :class:`pyxflow.Plot.xf_Plot`
                Plot handle
            *xfrac*: :class:`float`
                Fraction of horizontal portion of plot to fill

        :Returns:
            ``None``
        """
        # Versions:
        #  2013-12-15 @dalle   : First version

        # Get the axes handle.
        h_a = self.axes
        # Default xfraction
        if xfrac is None:
            # Check for a colorbar
            if self.colorbar is None:
                # Fill entire window
                xfrac = 1.0
            else:
                # Fill 90% of horizontal plot region.
                xfrac = 0.9
                # Move the colorbar.
                self.cax.set_position([0.91,0.03,0.1,0.94])
        # Set the position in relative (scaled) units.
        h_a.set_position([0, 0, xfrac, 1])
        # Update.
        if plt.isinteractive():
            plt.draw()

        return None
        

    # Method to draw a NEW figure at the correct size
    def AutoScale(self):
        """
        Alter the height of a plot window so that an XFlow plot has the proper
        aspect ratio
        
        This differs from the behavior of ``axis('equal')``, which changes the
        axis limits to preserve the one-to-one ratio instead of changing the
        figure size.
        
        :Call:
            >>> Plot.AutoScale()
        
        :Parameters:
            *Plot*: :class:`pyxflow.Plot.xf_Plot`
                Plot handle containing plot window to use
        
        :Returns:
            ``None``
        """

        # Versions:
        #  2013-12-15 @dalle   : First version

        # Extract the plot window limits.
        xmin = self.xmin
        xmax = self.xmax
        # Determine the dimension.
        dim = numel(xmin)
        # Only defined for two-dimensional plots.
        if dim == 3:
            # Not implemented!
            print "Not implemented for three-dimensional plots!"
            return None
        elif dim != 2:
            return None
            
        # Get the aspect ratio.
        AR = (xmax[1] - xmin[1]) / (xmax[0] - xmin[0])

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
        self.axes.set_xlim(xmin[0], xmax[0])
        self.axes.set_ylim(xmin[1], xmax[1])
        # Set the figure size (and update).
        self.figure.set_size_inches(w, h, forward=True)
        
    
    def HideBox(self):
        """
        Turn off right and top axes of the plot border
        
        :Call:
            >>> Plot.HideBox()
        
        :Parameters:
            *Plot*: :class:`pyxflow.Plot.xf_Plot`
                Plot handle to be updated
        
        :Returns:
            ``None``
        """
        # Use the method below.
        hide_box(self.axes)
        

    def Show(self):
        """
        Update or show the figure for an *xf_Plot* object
        
        :Call:
            >>> Plot.Show()
            
        :Parameters:
            *Plot*: :class:`pyxflow.Plot.xf_Plot`
                Plot handle to be updated or shown
        
        :Returns:
            ``None``
        """
        self.figure.show()

    # Method to customize the colormap
    def set_colormap(self, colorList):
        """
        Customize the colormap for an *xf_Plot* object
        
        :Call:
            >>> Plot.set_colormap(colorList)
        
        :Parameters:
            *Plot*: :class:`pyxflow.Plot.xf_Plot`
                Plot handle to have its colormap altered
            *colorList*: list of color descriptors
                List of colors or list of colors and break points
        
        :Returns:
            ``None``
        
        :See also:
            :func:`pyxflow.Plot.set_colormap()`
        """
        # Versions:
        #  2013-12-15 @dalle   : First version

        # Use the method below.
        if self.scalar is not None:
            set_colormap(self.scalar, colorList)
        # Check for a contour plot as well.
        if self.contour is not None:
            set_colormap(self.contour, colorList)
        return None


def set_colormap(h, colorList, vabs=None):
    """
    Apply a customized color map using a simple list of colors
    
    :Call:
        >>> set_colormap(h, colorList)
    
    :Parameters:
        *h*: Any matplotlib object with a *set_cmap* method
            Plot handle to have its colormap altered
        *colorList*: list of color descriptors
            List of colors or list of colors and break points
    
    :Returns:
        ``None``
        
    :Examples:
        The usual method is to apply a list of colors to an object with a
        colormap, for example a contour plot.  We begin by drawing a sample
        contour plot.
        
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> x = np.linspace(-3, 3, 61)
            >>> y = np.linspace(-2, 2, 41)
            >>> X, Y = np.meshgrid(x, y)
            >>> Z = np.sin(2*X) + np.cos(2*X) + 1.5*np.exp(-X*X-Y*Y)
            >>> h = plt.contourf(X, Y, Z)
        
        The first example changes the colormap to a monochrome scale with darker
        blues meaning low and lighter colors for high values.
        
            >>> pyxflow.set_colormap(h, ['Navy', 'b', (1,1,1)])
        
        Monochrome color themes are ideal for situations where the values are
        all positive.  In this case, some regions of the plot are negative, so
        it is helpful to use the colormap to highlight regions that are zero.
        
            >>> pyxflow.set_colormap(h, ['Navy', 'w', 'Maroon'])
            
        However, this plot turns out to have a maximum of about ``+3.0`` while
        the minimum is about ``-2.0``.  Therefore evenly spacing out these
        colors really highlights where the plot crosses the midpoint of this
        range, ``+0.5``.  We can fix this by specifying the _values_ we want the
        colors to occur at.
        
            >>> pyxflow.set_colormap(h, [(-2,'Navy'), (0,'w'), (3,'DarkRed')])
        
        This can also be done using relative values if you can determine them
        manually.  Thus the following gives nearly the same results.
        
            >>> pyxflow.set_colormap(h, [(0,'Navy'), (0.4,'w'), (1,'Maroon')])
        
        The function guesses which type of values you intended to use.  The only
        time this causes trouble is when you want to highlight absolute values
        between ``0`` and ``1``, but the limits of the contour plot go outside. 
        For this situation, use the following syntax.
        
            >>> pyxflow.set_colormap(h, [(0,'c'), (1,'Orange')], vabs=True)
            
        By repeating the same value twice in the color list, it is possible to
        dramatically highlight where the plot crosses a certain value.  This
        may be useful, for example, in the case of separating subsonic and
        supersonic flow.
        
            >>> h = plt.contourf(X, Y, Z, levels=np.linspace(-2,3,26)
            >>> pyxflow.set_colormap(h, [(-2,'r'),(0,'m'),(0,'c'),(2,'Navy')])
            
        Because Matplotlib's color converter is used, there are many ways to
        specify a color.  The following includes a red-green-blue tuple, a
        partially transparent color, and an HTML hash.
        
            >>> pyxflow.set_colormap(h, [(0.8,0.4,0.2), (0,0,1,0.5), '#0a8332'])
            
    """
    # Versions:
    #  2013-12-15 @dalle   : Introductory version

    # Determine the nature of the handle.
    if not hasattr(h, 'set_cmap'):
        raise AttributeError(
            "Input to `set_colormap` must have 'set_cmap' attribute.")

    # Check for named colormap
    if isinstance(colorList, str):
        # Pass it to set_cmap and see what happens.
        try:
            h.set_cmap(colorList)
            return None
        except:
            NotImplementedError(
                "Additional named colormaps have not been implemented.")

    # Get dimensions of the list.
    nColors = len(colorList)
    
    # Determine if break points are specified.
    if np.isscalar(colorList[0]) or len(colorList[0])!=2:
        # Default break points
        vList = np.linspace(0, 1, nColors)
        # Names of the colors
        cList = list(colorList)
    else:
        # User-specified break points
        vList = np.array([np.float(c[0]) for c in colorList])
        # Names of the colors
        cList = [c[1] for c in colorList]
        
    # Process the default scaling method if necessary.
    if vabs is None:
        if min(vList) == 0 and max(vList) == 1:
            # Scale the values from 0 to 1
            vabs = False
        else:
            # Use vList as absolute values.
            vabs = True
            
    # Scale the colors if requested.
    if vabs:
        # Set the color limits.
        h.set_clim(vList[0], vList[-1])
        # Re-scale the list to go from 0 to 1
        vList = (vList - vList[0]) / (vList[-1] - vList[0])
    
    # Convert the colors to RGB.
    rgba1List = colorConverter.to_rgba_array(cList)
    rgba2List = colorConverter.to_rgba_array(cList)
    # Find repeated colors.
    iRepeat = np.nonzero(np.diff(vList) <= 1e-8)
    # Compress the list of breakpoints.
    vList = np.delete(vList, iRepeat)
    # Process the repeated colors in reverse order.
    for i in iRepeat[::-1]:
        # Delete the second color at the same point from the "left" list.
        rgba1List = np.delete(rgba1List, i, 0)
        # Delete the first color at the same point from the "right" list.
        rgba2List = np.delete(rgba2List, i + 1, 0)
    
    # Make the lists for the individual components (red, green, blue).
    rList = [(vList[i], rgba2List[i, 0], rgba1List[i, 0])
        for i in range(vList.size)]
    gList = [(vList[i], rgba2List[i, 1], rgba1List[i, 1])
        for i in range(vList.size)]
    bList = [(vList[i], rgba2List[i, 2], rgba1List[i, 2])
        for i in range(vList.size)]
    aList = [(vList[i], rgba2List[i, 3], rgba1List[i, 3])
        for i in range(vList.size)]
    # Make the required dictionary for the colormap.
    cdict = {'red': rList, 'green': gList, 'blue': bList, 'alpha': aList}
    # Set the colormap.
    cm = LinearSegmentedColormap('custom', cdict)
    # Apply the colormap
    h.set_cmap(cm)
    # Update
    if plt.isinteractive():
        plt.draw()
    
    
# Function to turn off those fucking top and right axes
def hide_box(ax=None):
    """
    Hide right and top parts of plot box in one line
    
    :Call:
        >>> hide_box(ax)
        
    :Parameters:
        *ax*: :class:`matplotlib.axes.AxesSubplot`
            Handle to axes to be modified
    
    :Returns:
        ``None``
    """
    #Default input
    if ax is None:
        ax = plt.gca()
    # Delete the right and top edges of the plot.
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # Turn off the extra ticks.
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # Update.
    if plt.isinteractive():
        plt.draw()
    
    
# Function to give 0 for None, 1 for scalar, and number of elements for arrays
def numel(x):
    """
    Return number of elements in a list or ``1`` for a scalar or ``0`` for
    ``None``
    
    This function is still not nearly as useful as MATLAB's `numel` function.
    
    :Call:
        >>> n = numel(x)
    
    :Parameters:
        *x*: scalar, array, list, or ``None``
            A variable for which :func:`len` could be defined but might not be
    
    :Returns:
        *n*: int
            Number of elements in *x*
    """
    # Check for scalars (which have numel==1 for ... Python is so frustrating.
    if np.isscalar(x):
        return 1
    elif x is None:
        return 0
    else:
        return len(x)

# Function to process various descriptions of the plot bounding box
def GetXLims(Mesh, **kwargs):
    """
    Process a mesh and a list of optional arguments to create a bounding box.
    
    :Call:
        >>> xLimMin, xLimMax = GetXLims(Mesh, **kwargs)
        
    :Parameters:
        *Mesh*: :class:`pyxflow.Mesh.xf_Mesh`
            Instance of mesh to use for dimension count and default bounds
            
    :Returns:
        *xLimMin*: :class:`numpy.array` (``Mesh.Dim``)
            Minimum coordinate for each dimension
        *xLimMax*: :class:`numpy.array` (``Mesh.Dim``)
            Maximum coordinate for each dimension
            
    :Kwargs:
        *xmin*: float, array
            Minimum `x`-coordinate for plot window or array of minimum
            coordinates
        *xmax*: float, array
            Maximum `x`-coordinate for plot window or array of maximum
            coordinates
        *xlim*: array
            Minimum and maximum `x` coordinates or [`xmin`, `xmax`, `ymin`, ...]
        *ymin*: float
            Minimum `y`-coordinate for plot window
        *ymax*: float
            Maximum `y`-coordinate for plot window
        *ylim*: array
            List of [`ymin`, `ymax`]
        *zmin*: float
            Minimum `z`-coordinate for plot window
        *zmax*: float
            Maximum `z`-coordinate for plot window
        *zlim*: array
            List of [`zmin`, `zmax`]
        *xmindef*: array
            Default min coordinates to use instead of inspecting ``Mesh``
        *xmaxdef*: array
            Default max coordinates to use instead of inspecting ``Mesh``
    """
    # Check for default values not based on the mesh.
    xMinDef = kwargs.get('xmindef')
    xMaxDef = kwargs.get('xmaxdef')
    # Base off of mesh if necessary.
    if xMinDef is None:
        xMinDef = [Mesh.Coord[:,i].min() for i in range(Mesh.Dim)]
    if xMaxDef is None:
        xMaxDef = [Mesh.Coord[:,i].max() for i in range(Mesh.Dim)]
    # Initialize coordinate limits.
    xLimMin = [None for i in range(Mesh.Dim)]
    xLimMax = [None for i in range(Mesh.Dim)]
    # Lowest priority: list of xmins
    if kwargs.get('xmin') is not None:
        xmin = kwargs['xmin']
        # Check the dimensions.
        if numel(xmin) == Mesh.Dim:
            xLimMin = xmin
    # Lowest priority: list of xmins
    if kwargs.get('xmax') is not None:
        xmax = kwargs['xmax']
        # Check the dimensions.
        if numel(xmax) == Mesh.Dim:
            xLimMax = xmax
            
    # Next priority: full list
    if kwargs.get('xlim') is not None:
        xlim = kwargs['xlim']
        # Check the dimensions.
        if numel(xlim) == 2*Mesh.Dim:
            # Get every other element.
            xLimMin = xlim[0::2]
            xLimMax = xlim[1::2]
            
    # Second priority, individual limits
    if kwargs.get('xlim') is not None:
        xlim = kwargs['xlim']
        # Check the dimensions.
        if numel(xlim)==2 and Mesh.Dim>1:
            xLimMin[0] = xlim[0]
            xLimMax[0] = xlim[1]
    if kwargs.get('ylim') is not None:
        ylim = kwargs['ylim']
        # Check if it's appropriate.
        if numel(ylim)==2 and Mesh.Dim>1:
            xLimMin[1] = ylim[0]
            xLimMax[1] = ylim[1]
    if kwargs.get('zlim') is not None:
        zlim = kwargs['zlim']
        # Check if it's appropriate.
        if numel(zlim)==2 and Mesh.Dim>2:
            xLimMin[2] = zlim[0]
            xLimMax[2] = zlim[1]
            
    # Top priority: individual mins and maxes overrule other inputs.
    if kwargs.get('xmin') is not None:
        xmin = kwargs['xmin']
        # Check for a scalar (and relevance).
        if numel(xmin)==1 and Mesh.Dim>1:
            xLimMin[0] = xmin
    if kwargs.get('xmax') is not None:
        xmax = kwargs['xmax']
        # Check for a scalar (and relevance).
        if numel(xmax)==1 and Mesh.Dim>1:
            xLimMax[0] = xmax
    if kwargs.get('ymin') is not None:
        ymin = kwargs['ymin']
        # Check for a scalar.
        if numel(ymin)==1:
            xLimMin[1] = ymin
    if kwargs.get('ymax') is not None:
        ymax = kwargs['ymax']
        # Check for a scalar.
        if numel(ymax)==1:
            xLimMax[1] = ymax
    if kwargs.get('zmin') is not None:
        zmin = kwargs['zmin']
        # Check for a scalar.
        if numel(zmin)==1:
            xLimMin[2] = zmin
    if kwargs.get('zmax') is not None:
        zmax = kwargs['zmax']
        # Check for a scalar.
        if numel(zmax)==1:
            xLimMax[2] = zmax

    # Get the defaults based on all mesh coordinates.
    for i in range(Mesh.Dim):
        if xLimMin[i] is None:
            xLimMin[i] = xMinDef[i]
        if xLimMax[i] is None:
            xLimMax[i] = xMaxDef[i]
                
    # Output the limits.
    return xLimMin, xLimMax
                

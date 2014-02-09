
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

# Variable testing
from numpy import isscalar

# Class for xf_Plot objects
class xf_Plot:

    # Class initialization file.
    def __init__(self, All=None, Mesh=None, **kwargs):
        """
        Create a :class:`pyxflow.Plot.xf_Plot` object.
        
        :Call:
            >>> h_p = xf_Plot(All=None, Mesh=None, **kwargs)

        :Parameters:
            All: :class:`pyxflow.All.xf_All`
                Instance of XFlow all object containing mesh and solution
            Mesh: :class:`pyxflow.Mesh.xf_Mesh`
                Object used for plotting mesh without a solution

        :Returns:
            h_p: :class:`pyxflow.Plot.xf_Plot`
                Plot object with various properties

        :Kwargs:
            
            
            See also :func:`pyxflow.Plot.GetXLims`
        """
            
        # Extract the mesh.
        if All is not None and Mesh is None:
            Mesh = All.Mesh

        # Initialize some handles.
        self.figure = None
        self.axes = None
        self.mesh = None
        self.scalar = None
        self.contour = None
        # Store the limits.
        self.xmin = None
        self.xmax = None
        
        # Plot things if requested.
        # Note: Mesh=False hides the mesh if a valid All is given.
        if hasattr(Mesh, 'Coord'):
            Mesh.Plot()
        # Produce the initial plot.
        #self.Plot(All, **kwargs)
        #self.PlotMesh(Mesh, **kwargs)
    
    # Method to remove the plot (or parts of it).
    def remove():
        """
        
        """
        return None
        

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
    
    
# Function to give 1 for scalar and number of elements for arrays
def numel(x):
    """
    Return number of elements in a list or `1` for a scalar.
    
    This function is still not nearly as useful as MATLAB's `numel` function.
    
    :Call:
        >>> n = numel(x)
    
    :Parameters:
        x : float, int, array, list
            Any variable for which `len` should be defined but might not be
    
    :Returns:
        n : int
            Number of elements in `x`
    """
    # Check for scalars (which have numel==1 for ... Python is so frustrating.
    if isscalar(x):
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
        Mesh: :class:`pyxflow.Mesh.xf_Mesh`
            Instance of mesh to use for dimension count and default bounds
            
    :Returns:
        xLimMin: (`Mesh.Dim`) numpy.array
            Minimum coordinate for each dimension
        xLimMax: (`Mesh.Dim`) numpy.array
            Maximum coordinate for each dimension
            
    :Kwargs:
        xmin: float, array
            Minimum `x`-coordinate for plot window or array of minimum
            coordinates
        xmax: float, array
            Maximum `x`-coordinate for plot window or array of maximum
            coordinates
        xlim: array
            Minimum and maximum `x` coordinates or [`xmin`, `xmax`, `ymin`, ...]
        ymin: float
            Minimum `y`-coordinate for plot window
        ymax: float
            Maximum `y`-coordinate for plot window
        ylim: array
            List of [`ymin`, `ymax`]
        zmin: float
            Minimum `z`-coordinate for plot window
        zmax: float
            Maximum `z`-coordinate for plot window
        zlim: array
            List of [`zmin`, `zmax`]
        xmindef: array
            List of default min coordinates to use instead of inspecting `Mesh`
        xmaxdef: array
            List of default max coordinates to use instead of inspecting `Mesh`
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
                

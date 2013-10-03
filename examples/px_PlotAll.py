#!/usr/bin/python2
#
# Script to invoke <pyxflow.xf_All> `Plot` method on an '.xfa' file.
#
#   $ ./px_PlotAll.py in.xfa out.xfa
#
# Versions:
#  2013-10-01 @dalle   : First version


# Module to import command-line arguments.
import sys
# Add the pyxflow folder.
sys.path.append("..")
# PyPlot functions
import matplotlib.pyplot as plt
# Python/XFlow interface
import pyxflow as px


# Method
def main(argv):
    # Input file needed
    if len(argv) < 2:
        print "Usage:"
        print "  $ px_PlotAll.py in.xfa"
        print "  $ px_PlotAll.py in.xfa out.pdf"
        sys.exit(2)
    # Get the input file.
    ifile = argv[1]    
    # Default output file
    if len(argv) < 3:
        ofile = None
    else:
        ofile = argv[2]
    # Region
    xyrange = [-1, 2, -0.85, 1.41]
    # Read the file.
    All = px.xf_All(ifile)
    # Plot it.
    h_t = All.Plot(xyrange=xyrange)
    # Change the colormap.
    h_t.set_cmap('PRGn_r')
    # Set axes to equal.
    plt.axis("equal")
    # Restrict plot to requested window.
    plt.axis(xyrange)
    # Get the current axes handle
    h_a = plt.gca()
    # Fill the whole plot.
    h_a.set_position([0,0,1,1])
    # Turn off the ticks.
    h_a.set_xticks([])
    h_a.set_yticks([])
    # Update.
    plt.draw()
    # Now show it if no output file is given.
    if ofile is None:
        # Show the plot.
        plt.show()
    else:
        # Save the plot.
        plt.savefig(ofile, format="pdf")
        
if __name__ == "__main__":
    main(sys.argv)

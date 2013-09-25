"""File to interface with XFlow mesh objects in various forms"""

# Versions:
#  2013-09-25 @dalle   : First version

# ------- Class for xf_Geom objects -------
class xf_Geom:

	# Initialization method: can be read from '.geom' file or existing binary object
	def __init__(fname=None, ptr=None):

		
		# Versions:
		#  2013-09-24 @dalle   : First version

		# Check the parameters.
		if fname is not None:
			if ptr is not None:
				raise NameError
			# Read the file and get the pointer.
			ptr = px.ReadGeomFile(fname)
			# Set it.
			self._ptr = ptr
			self.owner = True
		elif ptr is not None:
			# Set the pointer.
			self._ptr = ptr
			self.owner = False
		else:
			# Create an empty geom.
			ptr = px.CreateGeom()
			# Set the pointer
			self._ptr = ptr
			self.owner = True
			# Exit the function
			return None

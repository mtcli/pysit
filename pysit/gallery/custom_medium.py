

import numpy as np

from pysit.gallery.gallery_base import GeneratedGalleryModel

from pysit import * #PML, Domain

__all__ = ['CustomMediumModel', 'custom_medium']

class CustomMediumModel(GeneratedGalleryModel):

    """ Gallery model for a generic, flat, layered medium. """

    model_name =  "Custom"

    valid_dimensions = (1,2,3)

    @property
    def dimension(self):
        return self.domain.dim

    supported_physics = ('acoustic',)

    def __init__(self, mesh,
                       C0_model,
                       dC_model=None,
                       **kwargs):
        """ Constructor for a custom velocity background model.  

        Parameters
        ----------
        mesh : pysit mesh
            Computational mesh on which to construct the model

        C0_model : function
            
        dC_model

        Notes
        -----
        * assumes C0_model() and dC_model() are compliant with dimensions and scaling
        of the computational mesh.

        """

        GeneratedGalleryModel.__init__(self)

        self._mesh = mesh
        self._domain = mesh.domain
        self.C0_model = C0_model
        self.dC_model = dC_model

        # Set _initial_model and _true_model
        self.rebuild_models()

    def rebuild_models(self):
        """ Rebuild the true and initial models based on the current configuration."""

        sh = self._mesh.shape(as_grid=True)
        grid = self._mesh.mesh_coords() # retrieve meshgrid

        if self.domain.dim==1:
            C0 = self.C0_model(grid[0])
        elif self.domain.dim==2:
            C0 = self.C0_model(grid[0], grid[1]).reshape(sh)
        elif self.domain.dim==3:
            C0 = self.C0_model(grid[0], grid[1], grid[2]).reshape(sh)

        self._initial_model = C0

        if self.dC_model is not None:
            if self.domain.dim==1:
                dC = self.dC_model(grid[0])
            elif self.domain.dim==2:
                dC = self.dC_model(grid[0], grid[1]).reshape(sh)
            elif self.domain.dim==3:
                dC = self.dC_model(grid[0], grid[1], grid[2]).reshape(sh)

            self._true_model = C0+dC
        else:
            self._true_model = C0

def custom_medium(mesh, C0_model, **kwargs):
    """ Friendly wrapper for instantiating the custom medium model. """

    model_config = dict(dC_model=None) # default keywords
    model_config.update(kwargs) 

    # Make any changes
    model_config.update(kwargs)

    return CustomMediumModel(mesh, C0_model, **model_config).get_setup()

#if __name__ == '__main__':
#
##  ASD = LayeredMediumModel(water_layered_rock)
##  ASD = LayeredMediumModel(water_layered_rock, initial_model_style='smooth', initial_config={'sigma':100, 'filtersize':150})
##  ASD = LayeredMediumModel(water_layered_rock, initial_model_style='gradient')
##  ASD = LayeredMediumModel(water_layered_rock, initial_model_style='constant', initial_config={'velocity':3000})
##  ASD = LayeredMediumModel(water_layered_rock, x_length=2000.0, y_length=1000.0)
#
#   C, C0, m, d = layered_medium(x_length=2000)
#
#   import matplotlib.pyplot as plt
#
#   fig = plt.figure()
#   fig.add_subplot(2,1,1)
#   vis.plot(C, m)
#   fig.add_subplot(2,1,2)
#   vis.plot(C0, m)
#   plt.show()

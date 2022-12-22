#! /usr/bin/env python3
#
# In (isogeometric) finite element simulations, locally refined meshes can be
# created using hierarchical meshes. In general, it is possible to construct
# meshes where neighboring elements may have a significant variation in size.
# When performing an analysis, however, such a size variation may not always
# be desirable. In that case, a mesh-regularization procedure, in which
# additional refinements are performed to reduce the size difference between
# neighboring elements, can be used.
#
# This example demonstrates a prototypical mesh regularization algorithm for
# hierarchical meshes. Although this algorithm is conceptually straightforward,
# it requires the advanced use of the Nutils element data structures (in
# particular the transformations that are used to identify elements). This
# example also aims to demonstrate how to use these data structures.

from nutils import mesh, function, cli, export, testing
import typing, numpy, matplotlib.collections

def main(nelems: int, nref: int, xref: typing.Tuple[float,float], difference: int):
    '''
    Regularization algorithm for a locally refined mesh.

    .. arguments::

       nelems [2]
         Number of elements along edge.
       nref [4]
         Number of refinement steps.
       xref [0.49,0.49]
         Point inside the element to be refined.
       difference [2]
         Maximum difference in refinement level between neighboring elements.
    '''

    # Construct the `nelems`x`nelems` base mesh.
    topo, geom = mesh.rectilinear([numpy.linspace(0,1,nelems+1)]*2)

    # Refine the element containing the `xref` point `nref` times.
    for iref in range(nref):

        # Get a (point) sample corresponding to the refinement point `xref`.
        refinement_point = topo.locate(geom, [[*xref]], tol=1e-12)

        # Evaluate the element index corresponding to the refinement point.
        elem_index = refinement_point.eval(topo.f_index)

        # Refine the element in which `xref` resides.
        topo = topo.refined_by(elem_index)

    # Plot the refined mesh.
    postprocess(topo, geom, f'refinement.png')

    # Call the mesh-regularization algorithm (implemented below).
    topo = regularize_mesh(topo, difference=difference)

    # Plot the regularized mesh.
    postprocess(topo, geom, f'regularized.png')

    # Return the sizes of the elements.
    return topo.sample('uniform', 1).eval(numpy.sqrt(function.J(geom)))

# Function implementing the mesh-regularization algorithm.
def regularize_mesh(topo, difference):

    # Refine the hierarhical mesh until satisfying the requirement on the
    # size difference between neighboring elements.
    while True:
        elem_indices = get_elements_to_be_refined(topo, difference)
        if not elem_indices:
            break
        topo = topo.refined_by(elem_indices)

    return topo

# Function returning the indices of elements that have a neighbour which
# is more than `difference` times refined compared to itself.
def get_elements_to_be_refined(topo, difference):
    assert difference > 0, 'difference must be positive'

    # Initiate the list of elements to be refined.
    elem_indices = []

    for transforms in topo.interfaces.transforms, topo.interfaces.opposites:
        for trans in transforms:
            index, tail = topo.transforms.index_with_tail(trans)
            if len(tail) > difference + 1:
                # Mark the element for refinement if the transformation from
                # interface to topology (consisting of one edge-to-volume
                # transformation and any number of coarsening transformations)
                # is longer than the specified allowable difference plus 1.
                elem_indices.append(index)

    return elem_indices

# Post-processing to plot the mesh and the refinement levels.
def postprocess(topo, geom, name):

    bezier = topo.sample('bezier',2)
    x, level = bezier.eval([geom, -numpy.log2(function.J(geom))/2])
    with export.mplfigure(name, dpi=150) as fig:
        ax  = fig.add_subplot(111, title='Refinement levels')
        im  = ax.tripcolor(x[:,0], x[:,1], bezier.tri, level, cmap='Accent')
        fig.colorbar(im)
        ax.add_collection(matplotlib.collections.LineCollection(x[bezier.hull,:2], colors='k', linewidths=1, alpha=1 if bezier.tri is None else .5))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_aspect('equal')

if __name__ == '__main__':
    cli.run(main)

# Regression test for the :mod:`nutils.testing` module.
class test(testing.TestCase):

    def test_baseline(self):
        element_sizes = main(nelems=2, nref=4, xref=(0.49,0.49), difference=2)
        with self.subTest('element-sizes'):
            self.assertAlmostEqual64(element_sizes, '''eNr7YTrXBB06GWODLwxBsNcAAgFHnRPN''')

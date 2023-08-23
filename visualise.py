import rdkit
print('rdkit version', rdkit.__version__)


import rdkit
from rdkit import Chem

from rdkit.Chem.Draw import IPythonConsole


# This script is referred from http://rdkit.blogspot.jp/2015/02/new-drawing-code.html
# and http://cheminformist.itmol.com/TEST/wp-content/uploads/2015/07/rdkit_moldraw2d_2.html
# from __future__ import print_function
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG
import pysvg
# import pysvg.structures
# import pysvg.builders
# import pysvg.text
import subprocess




import matplotlib.pyplot as plt
fig, ax = plt.subplots()

from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
def moltosvg(mol,molSize=(450,150),kekulize=True):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg

def render_svg(svg):
    # It seems that the svg renderer used doesn't quite hit the spec.
    # Here are some fixes to make it work in the notebook, although I think
    # the underlying issue needs to be resolved at the generation step
    return SVG(svg.replace('svg:',''))


smiles = 'O=C(NCC)COCc1cc(C(=O)F)c1N'

mol = Chem.MolFromSmiles(smiles)


mySvg = moltosvg(mol)
# pysvg.strcture.svg()
# savePathAndFile = "/myPath/testSvg.svg"
with open('pred_molecule.svg', 'w') as f:
    f.write(mySvg)


print('smiles:', smiles)
# display_mol = moltosvg(mol)


# display_mol.savefig('display_mol.svg', format='svg', dpi=1200)




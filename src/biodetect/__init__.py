"""biodetect: unsupervised detection of animal sounds in soundscape recordings.

A Python implementation of the segmentation and clustering pipeline of Guerrero et al.
(2023), "Acoustic animal identification using unsupervised learning", Methods in Ecology
and Evolution 14:1500-1514.

Modules
  segmentation   noise reduction, Otsu threshold, morphology, bounding boxes (2.3.1-2.3.2)
  speckle_null   label-free choice of a minimum box size (not in the paper)
  features       27 features per box (2.3.3)
  lamda          LAMDA clustering (2.3.4)
  clustering     features + LAMDA over a whole run
  figure         four-panel figure of one piece
  synthetic      synthetic soundscapes with known events
"""
__version__ = "0.1.0"

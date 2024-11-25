import os
# util.py: Utility functions

class FileSystemTree:
  def __init__(self, name, is_directory):
    self.children = []
    self.name = name
    self.is_directory = is_directory
  
  def add_child(self, child):
    self.children.append(child)

  def to_dict(self):
    """
    Convert the tree structure to a dictionary for easy JSON serialization.
    """
    if self.is_directory:
      return {
        'name': self.name,
        'is_directory': True,
        'children': [child.to_dict() for child in self.children]
      }
    else:
      return {
        'name': self.name,
        'is_directory': False
      }

# Use FileSystemTree class to store files/directories recursively
def buildFileTree(path):
  name = os.path.basename(path) or "None"
  tree = FileSystemTree(name, os.path.isdir(path))

  if os.path.isdir(path):
    try:
      for item in os.listdir(path):
        tree.add_child(buildFileTree(os.path.join(path, item)))
    except PermissionError:
      pass
  return tree

# Create 3D representation of the given mesh
# Feed this into three.js
def displayMesh(fileName):
  
  return
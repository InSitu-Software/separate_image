from setuptools import setup

setup(name='separate_image',
      version='0.1',
      description='separate image based on control elements',
      url='',
      author='David Lassner',
      author_email='davidlassner@gmail.com',
      packages=['separate_image'],
      install_requires=[
          'cv2',
          'numpy',
          'scipy'
      ],
      zip_safe=False)
# Semantic Human Texture Stitching

This repository contains texture stitching code corresponding to the paper **Detailed Human Avatars from Monocular Video**.

## Installation

```
pip install -r requirements.txt
```

Download and install [pyGCO](https://github.com/Borda/pyGCO).

## Usage

We provide sample scripts and data. All scripts output usage information when executed without parameters.

The software consists of three parts:

1. `step1_make_unwraps.py`: Creates partial textures.
2. `step2_segm_vote_gmm.py`: Creates semantic and GMM priors.
3. `step3_stitch_texture.py`: Stitches final texture.

### Quick start

```
bash run_sample.sh
```


## Data preparation

If you want to process your own data, some pre-processing steps are needed:

1. Run [PGN semantic segmentation](https://github.com/Engineering-Course/CIHP_PGN) on your input frames.
2. Run per-frame 3D pose detection plus SMPL offsets estimation e.g. by using [Octopus](https://github.com/thmoa/octopus).
3. Create the `frame_data.pkl` input file according to the following layout:

```
{
    "width": <image width>,
    "height": <image height>,
    "camera_f": <camera focal length>,
    "camera_c": <camera center>,
    "vertices": [<list of per frame SMPL vertices in camera coordinates>]
}
```

## Citation

This repository contains code corresponding to:

T. Alldieck, M. Magnor, W. Xu, C. Theobalt, and G. Pons-Moll.
**Detailed Human Avatars from Monocular Video**. In
*International Conference on 3D Vision*, IEEE, pp. 98-109, September 2018. 

Please cite as:

```
@inproceedings{alldieck2018detailed,
  title = {Detailed Human Avatars from Monocular Video},
  author = {Alldieck, Thiemo and Magnor, Marcus and Xu, Weipeng and Theobalt, Christian and Pons-Moll, Gerard},
  booktitle = {International Conference on 3D Vision},
  doi = {10.1109/3{DV}.2018.00022},
  pages = {98--109},
  month = {Sep},
  year = {2018}
}
```


## License

Copyright (c) 2019 Thiemo Alldieck, Technische Universität Braunschweig, Max-Planck-Gesellschaft

**Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").**

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the **Detailed Human Avatars from Monocular Video** paper in documents and papers that report on research using this Software.

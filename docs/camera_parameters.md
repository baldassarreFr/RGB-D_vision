# D435i camera model: parameter table

**The single reference for the camera model.** Every number in `src/rgbd_sim/d435.py`
appears here with its source. Discussion of each source: the report, Section 3.

Reference camera: **Intel RealSense D435i**, depth stream **848 x 480** (depth module D430).
Rule: official Intel/RealSense documents first; otherwise papers that measured the same
depth hardware (D435 or D435i: Intel lists D430/D435/D435i with identical depth
specifications [S4]). Nothing is invented: an effect without a measured value is left out of the camera model
(two of them are studied as a sensitivity in Step 2, see section 2).

## 1. Values used

| # | Quantity | Value | Source |
|---|---|---|---|
| 1 | Resolution | 848 x 480 | optimal D435 depth resolution [S3]; used by [R1] |
| 2 | Horizontal field of view | 87 deg (-> fx = fy = 446.8 px, square pixels) | D430 module table [S1]; D435i page "H:87 V:58" [S2] |
| 3 | Baseline (left to right imager) | 50 mm, right imager on the +x side | [S1]; depth referenced to the left imager, invalid band on the left [S1 sec. 4.5]; also [R3] |
| 4 | Subpixel accuracy of the matcher | 0.08 px | [S3] ("less than 0.1"; curves use 0.08) |
| 5 | Base noise (matte surface, seen face-on) | sigma_z = z^2 x 0.08 / (fx x b) -> 0.9 / 1.5 / 3.0 mm at 0.49 / 0.65 / 0.92 m | formula [S3]; about 1-2 mm at 60-70 cm measured with our exact settings [R1, Fig. 5] |
| 6 | Material factor: aluminium (parts) | x 2.9 (range 2.5 - 3.3 across lighting) | [R2], Fig. 6, D435 at 1 m, aluminium vs foam/paper/plastics, read from the figure |
| 7 | Material factor: table | x 1 (matte work surface) | **scene choice**, not a camera parameter |
| 8 | Edge artifacts: size | edges displaced by up to 1 px | [R2], Table 2: D435 lateral noise 1 px at 0 - 3 m (maximum distance of edge pixels from the true straight edge) |
| 9 | Edge artifacts: how often | each pixel on an object outline takes its neighbour's depth with probability 1/2 | **our choice**: [R2] gives the maximum displacement only; 1/2 = edge equally likely to move either way |

Numerical tolerance in `camera_model.py` (geometry bookkeeping, **not** a camera parameter):
a point counts as seen by the right imager unless something is more than 1 mm in front of it
(stored depth is rounded to 0.1 mm). Edges are found from the labels (outline of each object),
with no threshold.

## 2. Left out, and why

| Effect | Reason |
|---|---|
| Noise growing with the incidence angle | the only model for this camera, Ahn et al. Eq. 2 [R3] (fitted on a white wall, 0.4 - 2.1 m, 0 - 75 deg; angle term "heuristically obtained"), gives on our scene 0 a median sigma of **44 mm** on part faces seen at >= 45 deg (128 mm with the aluminium factor), i.e. more than the size of the parts. This contradicts object-level measurements with the same camera at 60 cm (mean Chamfer distance about 0.6 - 0.9 cm, [R1] Fig. 14). No reliable value -> not modelled (checks/check_noise.py, 4 Nov) |
| Mixed depths at edges ("flying pixels" between foreground and background) | no measurement for this camera; edges are only displaced, never blended |
| Edges inside one object (rim of a slot, one bracket leg in front of the other) | simplification: only object outlines move (edges found from labels, no threshold); slots still lose depth through the stereo shadows |
| Range limits (min ~0.17-0.3 m, max ~3 m) | never reached: scene depth is 0.49 - 0.92 m [S2, S3] |
| Holes / glare on aluminium at oblique angles | no measurement for this camera: [R2] tested materials face-on only and lists angles as future work. **Step 2** studies holes at critical angles of 75, 45 and 30 deg (`camera_model.aluminium_holes_mask`) |
| Holes on aluminium in very bright light | seen in [R2] (no value at 535 lux, 1 m) but not explained and not seen at 1.5 m; mentioned, not modelled |
| Smoothing of thin features | depends on the matching window, which Intel does not publish [R4] |
| Wrong depth from highlights / inter-reflections | no measurement; needs a reflection-aware renderer |
| Projector shadows | projector sits between the imagers [S6], so its shadows lie inside the stereo shadows |
| Spatial correlation of the noise (errors in neighbouring pixels move together) | real: the matcher works on windows and aggregates neighbours [R4]; [R1] reports depth "layers"; but the size of the correlated patches is not published or measured for this camera -> noise is independent per pixel. **Step 2** studies correlation lengths of 2, 5 and 10 px (`camera_model.correlated_field`) |
| Temporal noise, calibration bias, post-processing filters | single frames from a calibrated camera, no post-processing (as in [R1]) |

## 3. Sources

- [S1] RealSense, *D400 Series Product Family Datasheet*, Mar 2026. https://www.realsenseai.com/wp-content/uploads/2026/03/RealSense-D400-Series-Datasheet-Mar-2026.pdf
- [S2] Intel, *RealSense Depth Camera D435i - specifications*. https://www.intel.com/content/www/us/en/products/sku/190004/intel-realsense-depth-camera-d435i/specifications.html
- [S3] A. Grunnet-Jepsen, J. N. Sweetser, J. Woodfill, *Tuning depth cameras for best performance*, RealSense white paper, rev. 2.0. https://dev.realsenseai.com/?p=41439
- [S4] RealSense support, *Depth accuracy for D400 series cameras*. https://support.realsenseai.com/hc/en-us/articles/360059129453-Depth-accuracy-for-Intel-RealSense-D400-Series-Cameras
- [S6] *White paper on projectors for RealSense D4xx*. https://realsenseai.com/wp-content/uploads/2019/03/WhitePaper_on_Projectors_for_RealSense_D4xx_1.0.pdf
- [R1] L. Rustler, V. Volprecht, M. Hoffmann, "Empirical Comparison of Four Stereoscopic Depth Sensing Cameras for Robotics Applications," IEEE Access 13:67564-67577, 2025. doi:10.1109/ACCESS.2025.3560810
- [R2] G. Halmetschlager-Funek, M. Suchi, M. Kampel, M. Vincze, "An Empirical Evaluation of Ten Depth Cameras ...," IEEE Robotics & Automation Magazine 26(1):67-77, 2019. doi:10.1109/MRA.2018.2852795
- [R3] M. S. Ahn, H. Chae, D. Noh, H. Nam, D. Hong, "Analysis and Noise Modeling of the Intel RealSense D435 for Mobile Robots," Int. Conf. on Ubiquitous Robots (UR), 2019, pp. 707-711. doi:10.1109/URAI.2019.8768489
- [R4] L. Keselman, J. Iselin Woodfill, A. Grunnet-Jepsen, A. Bhowmik, "Intel RealSense Stereoscopic Depth Cameras," CVPR Workshops 2017. arXiv:1705.05548

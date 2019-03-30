# SeamCarving
[ITCS381]SeamCarving Project

### Note
**Horizontal Seam insertion/removal** - counter clockwise rotate image 90 degree then insert/remove seam.

##### Dificulty: Intermediate (medium)

### Challenges
**M & K matrix computation** - this process take time because we can't spot the overflow error from addition, so the result of M matrix is look abnormal when compare to the image in the slide. To fix this problem, we create M matrix with CV_32S type or integer type to prevent the overflow, and then convert it back to CV_8U type or unsigned char type using normalization.

**Seam insertion** - this is process is quite complicated because of best seam computation. If insert seam normally and then use seam inserted image to compute M & K matrix. The result will be the same seam as a previous one. To fix this problem, A duplicated image is require. Duplicated image get a seam removal, while original image get a seam insertion, and use duplicated image to compute M & K matrix instead of seam inserted one. So, the result will be a new seam.

### Possible Future Developments
* Increase the performance and speed of seam insertion/removal
* Insert/remove many seam at once.
* Object Retaining
* Object Removal
* Mouse Interaction

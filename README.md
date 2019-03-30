# SeamCarving
[ITCS381]SeamCarving Project

### Note
**Seam insertion** - this is process is quite complicated because of best seam computation. If insert seam normally and then use seam inserted image to compute M & K matrix. The result will be the same seam as a previous one. To fix these problem. A duplicated image is require. Duplicated image get a seam removal, while original image get a seam insertion, and use duplicated image to compute M & K matrix instead of seam inserted one. So, the result will be a new seam.

**Horizontal Seam insertion/removal** - counter clockwise rotate image 90 degree then insert/remove seam.

### Dificulty: Intermediate (medium)

### Challenges

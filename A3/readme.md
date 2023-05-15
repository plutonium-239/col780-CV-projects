<center><h1> COL780 Assignment 3 </h1></center>

<div align='right'> <b> Samarth Bhatia <br> 2019CH10124 </b> </div>

# Part 1: Intrinsic Camera Parameter Calculation

### Details/Formulation

I will be using Zhang's method to perform calibration (using a checkerboard pattern) to calculate intrinsic parameters for a camera projecting 3d world coordinates to 2d image coordinates.

Note that I have avoided using `cv2` functions unless necessary: I don't use even the `findChessboardCorners` function and use a non-standard checkerboard pattern. 

Let $[x,y,z]$ be a point in the 3d space, and its corresponding point in the image be $[u,v]$. Then, we write:
$$
\begin{aligned}
\begin{bmatrix}u\\ v\\ 1\end{bmatrix} &= 
\begin{bmatrix}m_{11} & m_{12} & m_{13} & m_{14}\\ m_{21} & m_{22} & m_{23} & m_{24}\\ m_{31} & m_{32} & m_{33} & m_{34}\end{bmatrix}
\begin{bmatrix}x\\ y\\ z\\ 1\end{bmatrix}
\\
P &= MX
\end{aligned}
$$
However, this does not separate the intrinsic parameters (that map the camera 3d coordinates to the 2d image plane) from the extrinsic parameters (that map 3d world coordinates to 3d camera coordinates).

We can write this matrix as:
$$
\label{eq:2}
\begin{aligned}
\begin{bmatrix}u\\ v\\ 1\end{bmatrix} &= 
\begin{bmatrix}f\cdot m_x & \gamma & v_0\\ 0 & f\cdot m_y & u_0\\ 0 & 0 & 1\end{bmatrix}
\begin{bmatrix}r_{11} & r_{12} & r_{13} & t_1\\ r_{21} & r_{22} & r_{23} & t_2\\ r_{31} & r_{32} & r_{33} & t_3\end{bmatrix}
\begin{bmatrix}x\\ y\\ z\\ 1\end{bmatrix}
\\
P &= K[R|t]X
\end{aligned}
$$
Here $K$ is the intrinsic parameter matrix. $f$ is the focal length, $m_x,m_y$ are pixel densities of the sensor in x and y direction, $\gamma$ is the skew in x-direction, and $(u_0,v_0)$ is the image of the principal point.

If we take a flat checkerboard pattern, we can set the 3d coordinate system to be at one of the corners in the plane of the pattern. This gives us that $z=0$ for all points on the checkerboard, and their $x,y$ coordinates can be easily calculated.
So, putting $z=0$ in $\ref{eq:2}$, we can ignore the 3^rd^ column of [R|t], the extrinsic parameter matrix. So, our eqn becomes:
$$
\begin{aligned}
\begin{bmatrix}u\\ v\\ 1\end{bmatrix} &= 
\begin{bmatrix}f\cdot m_x & \gamma & v_0\\ 0 & f\cdot m_y & u_0\\ 0 & 0 & 1\end{bmatrix}
\begin{bmatrix}r_{11} & r_{12} & t_1\\ r_{21} & r_{22} & t_2\\ r_{31} & r_{32} & t_3\end{bmatrix}
\begin{bmatrix}x\\ y\\ 1\end{bmatrix}
\\ &\ or,\\ 
P &= HX
\end{aligned}
$$
So, we find multiple correspondences on the checkerboard in the 2d and 3d coordinates $(x_i,y_i,0) -> (u_i,v_i)$, and use them to find a $3\times 3$ homography instead of a $3\times 4$ projection. We need 4 points to determine H.
**Note that intrinsic parameters are same for all checkerboard points in all images but extrinsic parameters are only same for a single frame.** 
$$
h = \begin{bmatrix}H_{\cdot1} & H_{\cdot3} & H_{\cdot2}\end{bmatrix}^T \\
a^T_{u,i} = \begin{bmatrix} -x_i & -y_i & -1 &0&0&0& u_ix_i & u_iy_i & u_i\end{bmatrix}\\
a^T_{v,i} = \begin{bmatrix} 0&0&0& -x_i & -y_i & -1 & v_ix_i & v_iy_i & v_i\end{bmatrix}
$$


Since $H$ is not a product of an upper triangular matrix and a rectangular matrix anymore (as in the Direct Linear Transform method), we can't use QR factorization to get $K$ and $\mathbf{[r_1, r_2,t]}$. So, we make use of the fact that:

1. $K$ is invertible (as it is an upper-triangular matrix and has non-zero diagonal elements) $\mathbf{r_1} = K^{-1}\mathbf{h_1}$ and $\mathbf{r_2} = K^{-1}\mathbf{h_2}$
2. $\mathbf{r_1}$ and $\mathbf{r_2}$ are orthogonal to each other (as they are part of a rotation matrix), $\mathbf{r_1^T r_2^\ = 0}$
3. $||r_1|| = ||r_2|| = 1$

These give us:
$$
\label{eq:4}
\begin{aligned}
\mathbf{h_1^T}B\mathbf{h_2} &= 0\\
\mathbf{h_1^T}B\mathbf{h_1} - \mathbf{h_2^T}B\mathbf{h_2} &= 0 \\
\text{where,}\ B &= (K^{-1})^TK^{-1}\ \text{is positive definite}
\end{aligned}
$$
Since $B$ is symmetric, we only have 6 unknowns (consider the upper triangle), $\mathbf{b} = [b_{11}, b_{12}, b_{13}, b_{22}, b_{23}, b_{33}]$.
We can write $\ref{eq:4}$ in a linear form:
$$
\begin{aligned}
\mathbf{v_{12}^T b = 0}\ &\text and\ \mathbf{(v_{11}^T - v_{22}^T) b = 0}\\
\text where,&\\
\mathbf{v_{ij}} =& 
	\begin{bmatrix}
	h_{1i}h_{1j} \\ h_{1i}h_{2j} + h_{2i}h_{1j} \\ h_{1i}h_{3j} + h_{3i}h_{1j} \\ h_{2i}h_{2j} \\
	h_{2i}h_{3j} + h_{3i}h_{2j} \\ h_{3i}h_{3j}
	\end{bmatrix}
\\
\text or,&\\
\begin{bmatrix}\mathbf{v_{12}^T} \\ \mathbf{v_{11}^T - v_{22}^T}\end{bmatrix}\mathbf{b} =&\ \mathbf{0} \\

\end{aligned}
$$
This only gives us 2 equations, so we do the same procedure for multiple images, say $n \geq 3$, and stack them together - like we stack multiple points together to find transformations. Also, we add a constraint $||\mathbf{b}||=1$ to remove the trivial solution $\mathbf{b = 0}$. This gives us:
$$
\begin{aligned}
\begin{bmatrix}
\mathbf{v1_{12}^T} \\ \mathbf{v1_{11}^T - v1_{22}^T} \\ \vdots \\ \mathbf{vn_{12}^T} \\ \mathbf{vn_{11}^T - vn_{22}^T}
\end{bmatrix}\mathbf{b} &= \mathbf{0} \\
V\mathbf{b} &= \mathbf{0}\\
\text or \\

\min_{\mathbf{b}}&\ ||V\mathbf{b}|| \\
s.t.&\ ||\mathbf{b}||=1 
\end{aligned}
$$
So, to minimize this (or find a vector in the null-space of V, $\mathcal{N}(V)$), we find the SVD of V. Let $V = MSN^T$ be the SVD where $M$ is $2n\times6$, and $S$ and $N$ are $6\times6$.

Then, we choose $\mathbf{b} = \mathbf{n_6}$ (the singular vector belonging to smallest singular value of $V$). Rearrange this to find $B$.
Now, if we find the cholesky decomposition of $B$,
$$
\text{chol}(B) = AA^T. \\
\text{Clearly, from \ref{eq:4},}\ A = (K^{-1})^T
$$
 Now we have $K$, the required intrinsic camera parameters.

### Results

The camera matrix I obtained is (using 3 images):
$$
\begin{bmatrix}
	0.5406 & 0.3345 & 765.37\\
 	0 & 0.6653 & 214.96\\
 	0 & 0 & 1
\end{bmatrix}
$$
**Note that the values are different in magnitude because I have chosen the points in cm.**

Comparing to the original focal length of my phone's camera[^1],[^2], we are not far off from the actual value ($\approx$0.39). And the center of the image is also correct (image dimensions were $2016\times930$). Note that using more than 3 images, our objective will try to minimize $||V\mathbf{b}||$ and so the translation column will be wrong (i.e. the last singular value is not the right solution anymore).

However, we are getting a *skew* parameter of 0.3345, which is not ideal, and we are limited by the SVD and the Cholesky decomposition to be able to choose a more accurate mathematical model of the $K$ matrix.

# Part 2: Placing 3D objects in the image (AR)

For this part, there is not much math involved. However, we do need to find an estimate for the projection matrix from the camera parameters and the homographies.
$$
\def\mat#1{\begin{bmatrix}#1\end{bmatrix}} 
\begin{aligned}
KH &= \mat{R&|&t} \\ 
R &= \mat{R_1& R_2} \\
\end{aligned}
$$
Now, we might not find $R_1$ and $R_2$ that are orthogonal, and we need to find $R_1, R_2, R_3$ s.t. they are orthonormal. We can simply:
$$
R_1' = R_1 + R_2\\
R_2' = R_1 \times R_2\\
\implies R_1' \times R_2' = 0\ \text{[orthogonal]}\\

\text{Take}\ R_3' = R_1'\times R_2' \implies \text{orthonormal vectors}\\

P = \mat{R_1'& R_2'& R_3'& t}
$$
Now, we can easily project points from 3D to 2D by 
$$
\mat{u\\v\\1} = \underset{3\times3}K\ \ \underset{3\times4}P\mat{X\\Y\\Z\\1}
$$
Now that we have the pixel coordinates, we can simply fill the pixels with the correct colors and we have ourselves an AR image!

### Results

*I was not able to get loading of textures for 3D models working, so the color functionality does not work for loaded models. However, there is enough detail in the images to tell that the code is working and the objects are situated on the plane.*

**Simple Cube**

Blue plane is the roof and pink plane is the base. It is positioned at the corner from where (0,0) was measured on the chessboard. This works very well. This example idea (of just taking 8 points of a cube and transforming them) is available on OpenCV[^3].

![image-20230407173226027](https://i.ibb.co/4S8rtng/image-20230407173226027.png)

![image-20230407172817597](https://i.ibb.co/hB7kHzZ/image-20230407172817597.png)

![image-20230407172851277](https://i.ibb.co/TRy03s2/image-20230407172851277.png)

![image-20230407172900478](https://i.ibb.co/QFQsb06/image-20230407172900478.png)

**Simple Prism**

3D Model:

![image-20230331200351054](https://i.ibb.co/p1Nkgqt/image-20230331200351054.png)

Projected into images:

![image-20230331200043972](https://i.ibb.co/yNBzsHR/image-20230331200043972.png)

![image-20230331200421107](https://i.ibb.co/G2x4sR6/image-20230331200421107.png)

![image-20230331200453067](https://i.ibb.co/cJw5zQT/image-20230331200453067.png)

![image-20230331200506567](https://i.ibb.co/xz5sHJJ/image-20230331200506567.png)

*Even with a change in angle, the projected prism has one face with the diagonal of the checkerboard as a normal, just like in previous images.*

**With a car**

3D Model:

![image-20230331200919759](https://i.ibb.co/b2rJj33/image-20230331200919759.png)

Projected into images:

![image-20230331200745265](https://i.ibb.co/2kKQkmt/image-20230331200745265.png)

![image-20230331200945743](https://i.ibb.co/PGKWwG7/image-20230331200945743.png)

![image-20230331200952811](https://i.ibb.co/k28H1KN/image-20230331200952811.png)

# Usage

```python
usage: main.py [-h] [--color] [--model MODEL]

optional arguments:
  -h, --help     show this help message and exit
  --color        To give colors to faces
  --model MODEL  Which model to show: -1=Simple Cube (default) 0=Prism 1=Cybertruck 2=Building 3=Katana
  --scale SCALE  Scale of model (default=3)
```

`main.py` contains the driver code to run everything.
`find_intrinsic_params.py` contains the code for Part 1.
`overlay_object.py` contains the code for Part 2.

# References

[^1]: https://www.camerafv5.com/devices/manufacturers/samsung/sm-a505g_a50_0/
[^2]: https://learnopencv.com/approximate-focal-length-for-webcams-and-cell-phone-cameras/
[^3]: https://docs.opencv.org/3.4/d7/d53/tutorial_py_pose.html

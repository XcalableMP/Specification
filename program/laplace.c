/*
 *  A parallel explicit solver of Laplace equation in \XMP
 */
#pragma xmp nodes p(NPROCS)
#pragma xmp template t(1:N)
#pragma xmp distribute t(block) onto p

double u[XSIZE+2][YSIZE+2],
       uu[XSIZE+2][YSIZE+2];
#pragma xmp align u[i][*] to t(i)
#pragma xmp align uu[i][*] to t(i)
#pragma xmp shadow uu[1:1][0:0]

lap_main()
{ 
 int x,y,k;
 double sum;
for(k = 0; k < NITER; k++){
	/* old <- new */
#pragma xmp loop on t(x)
	for(x = 1; x <= XSIZE; x++)
	  for(y = 1; y <= YSIZE; y++)
	    uu[x][y] = u[x][y];
#pragma xmp reflect (uu)
#pragma xmp loop on t(x)
	for(x = 1; x <= XSIZE; x++)
	  for(y = 1; y <= YSIZE; y++)
	    u[x][y] = (uu[x-1][y] + uu[x+1][y] +
                  uu[x][y-1] + uu[x][y+1])/4.0;
    }
m */
    sum = 0.0;
#pragma xmp loop on t[x] reduction(+:sum)
    for(x = 1; x <= XSIZE; x++)
	for(y = 1; y <= YSIZE; y++)
	  sum += (uu[x][y]-u[x][y]);
#pragma xmp task on p(1)
    printf("sum = %g\n",sum);
}

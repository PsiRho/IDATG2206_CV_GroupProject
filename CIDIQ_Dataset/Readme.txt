The Colourlab Image Database:Image Quality (CID:IQ) contains a set of images and their corresponding subjective scores developed by the Norwegian Colour and Visual Computing Laboratory at Gjøvik University College. For more information visit www.colourlab.no/cid

If you use this work please cite: 
Xinwei Liu, Marius Pedersen and Jon Yngve Hardeberg. CID:IQ - A New Image Quality Database. International Conference on Image and Signal Processing (ICISP 2014). Vol. 8509. Cherbourg, France. Jul., 2014.

Colourlab Image Database:Image Quality. Url: www.colourlab.no/cid. 2014. 


-----------------------------------------------------------
The database contains 23 pictorial reference images
Six distortions are applied to all reference images:
1. JPEG compression 
2. JPEG2000 compression 
3. Poisson noise
4. Gaussian blur 
5. ?E gamut mapping 
6. SGCK gamut mapping

All reference images are applied these distortions in five levels from low degree of quality to high degree of quality degradation.

Ambient illumination for the experiment was approximately 4 lux. The chromaticity of the white is D65 and luminance level is 80 cd/m2. All settings are suited for sRGB color space.

The observers viewed the images in two separate session at different viewing distances (viewing angle of 23 degrees) and 100cm (viewing angle of 12 degrees) 

17 human observers participated the psychometric experiment. Category judgment (9 categories) was used as the scaling approach.


MATLAB code
Together with the database a matlab code to calculate image quality metrics is provided: RunMetrics.m. In this code you can easily add the image quality metrics you want to test. 

Calculation of Pearson, Spearman, and Kendall correlation coefficients and RMSE is done with the code CorrCIDIQ.m. This code will also generate barplots and latex tables. 

We do not guaranttee the code to work, and if you find any problems or have any questions related to CID:IQ please contact Dr. Marius Pedersen (marius.pedersen@hig.no). 

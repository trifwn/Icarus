READ THE FLOW AND GEOMETRICAL DATA FOR EVERY SOLID BODY
               <blank>
               <blank>
Body Number   NB = 1
               <blank>
2           NLIFT   
0           IYNELSTB   
1           NBAER2ELST 
9           NNBB    
15          NCWB    
2           ISUBSCB
2
3           NLEVELSB
1           IYNTIPS 
0           IYNLES  
0           NELES   
0           IYNCONTW
3           IDIRMOB  direction for the torque calculation
               <blank>
4           LEVEL  the level of movement
               <blank>
Give  data for every level
NB=1, lev=3  (pitch )
Rotation
1           IMOVEAB  type of movement
2           NAXISA   =1,2,3 axis of rotation
-0.000001   TMOVEAB  -1  1st time step
100.        TMOVEAB  -2  2nd time step
0.          TMOVEAB  -3  3d  time step
0.          TMOVEAB  -4  4th time step!---->omega
   2.0      AMOVEAB  -1  1st value of amplitude
   2.0      AMOVEAB  -2  2nd value of amplitude
0.          AMOVEAB  -3  3d  value of amplitude
0.          AMOVEAB  -4  4th value of amplitude!---->phase
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
Translation
0           IMOVEUB  type of movement
2           NAXISU   =1,2,3 axis of translation
-0.000001   TMOVEUB  -1  1st time step
10.         TMOVEUB  -2  2nd time step
0.          TMOVEUB  -3  3d  time step
0.          TMOVEUB  -4  4th time step
   0.       AMOVEUB  -1  1st value of amplitude
   0.       AMOVEUB  -2  2nd value of amplitude
0.          AMOVEUB  -3  3d  value of amplitude
0.          AMOVEUB  -4  4th value of amplitude
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
NB=1, lev=2  (roll)
Rotation
1           IMOVEAB  type of movement
1           NAXISA   =1,2,3 axis of rotation
-0.000001   TMOVEAB  -1  1st time step
10.         TMOVEAB  -2  2nd time step
0.          TMOVEAB  -3  3d  time step
0.          TMOVEAB  -4  4th time step
   1.5       AMOVEAB  -1  1st value of amplitude
   1.5       AMOVEAB  -2  2nd value of amplitude
0.          AMOVEAB  -3  3d  value of amplitude
0.          AMOVEAB  -4  4th value of amplitude
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
Translation
0           IMOVEUB  type of movement
1           NAXISU   =1,2,3 axis of translation
0.          TMOVEUB  -1  1st time step
0.          TMOVEUB  -2  2nd time step
0.          TMOVEUB  -3  3d  time step
0.          TMOVEUB  -4  4th time step
0.          AMOVEUB  -1  1st value of amplitude
0.          AMOVEUB  -2  2nd value of amplitude
0.          AMOVEUB  -3  3d  value of amplitude
0.          AMOVEUB  -4  4th value of amplitude
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
NB=1, lev=1  ( yaw     )
Rotation
1           IMOVEAB  type of movement
3           NAXISA   =1,2,3 axis of rotation
-0.000001   TMOVEAB  -1  1st time step
10.         TMOVEAB  -2  2nd time step
0.          TMOVEAB  -3  3d  time step
0.          TMOVEAB  -4  4th time step
   5.0      AMOVEAB  -1  1st value of amplitude  (Initial MR phase + 0)
   5.0      AMOVEAB  -2  2nd value of amplitude          
0.          AMOVEAB  -3  3d  value of amplitude
0.          AMOVEAB  -4  4th value of amplitude
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
Translation
0           IMOVEUB  type of movement
3           NAXISU   =1,2,3 axis of translation
0.          TMOVEUB  -1  1st time step
0.          TMOVEUB  -2  2nd time step
0.          TMOVEUB  -3  3d  time step
0.          TMOVEUB  -4  4th time step
0.          AMOVEUB  -1  1st value of amplitude
0.          AMOVEUB  -2  2nd value of amplitude
0.          AMOVEUB  -3  3d  value of amplitude
0.          AMOVEUB  -4  4th value of amplitude
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
-----<end of movement data>----------------------------------------------------
               <blank>
Cl, Cd data / IYNVCR(.)=0 then Cl=1., Cd=0.
1           IYNVCR(1)
name.cld     FLCLCD      file name wherefrom Cl, Cd are read
               <blank>
Give the file name for the geometrical distributions
Lwing.bld 
               <blank>
Body Number   NB = 2
               <blank>
2           NLIFT   
0           IYNELSTB   
2           NBAER2ELST 
9           NNBB    
15          NCWB    
2           ISUBSCB
2
3           NLEVELSB
1           IYNTIPS 
0           IYNLES  
0           NELES   
0           IYNCONTW
3           IDIRMOB  direction for the torque calculation
               <blank>
4           LEVEL  the level of movement
               <blank>
Give  data for every level
NB=2, lev=3  (pitch )
Rotation
1           IMOVEAB  type of movement
2           NAXISA   =1,2,3 axis of rotation
-0.000001   TMOVEAB  -1  1st time step
100.        TMOVEAB  -2  2nd time step
0.          TMOVEAB  -3  3d  time step
0.          TMOVEAB  -4  4th time step!---->omega
   2.0      AMOVEAB  -1  1st value of amplitude
   2.0      AMOVEAB  -2  2nd value of amplitude
0.          AMOVEAB  -3  3d  value of amplitude
0.          AMOVEAB  -4  4th value of amplitude!---->phase
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
Translation
0           IMOVEUB  type of movement
2           NAXISU   =1,2,3 axis of translation
-0.000001   TMOVEUB  -1  1st time step
10.         TMOVEUB  -2  2nd time step
0.          TMOVEUB  -3  3d  time step
0.          TMOVEUB  -4  4th time step
   0.       AMOVEUB  -1  1st value of amplitude
   0.       AMOVEUB  -2  2nd value of amplitude
0.          AMOVEUB  -3  3d  value of amplitude
0.          AMOVEUB  -4  4th value of amplitude
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
NB=2, lev=2  (roll)
Rotation
1           IMOVEAB  type of movement
1           NAXISA   =1,2,3 axis of rotation
-0.000001   TMOVEAB  -1  1st time step
10.         TMOVEAB  -2  2nd time step
0.          TMOVEAB  -3  3d  time step
0.          TMOVEAB  -4  4th time step
   1.5       AMOVEAB  -1  1st value of amplitude
   1.5       AMOVEAB  -2  2nd value of amplitude
0.          AMOVEAB  -3  3d  value of amplitude
0.          AMOVEAB  -4  4th value of amplitude
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
Translation
0           IMOVEUB  type of movement
1           NAXISU   =1,2,3 axis of translation
0.          TMOVEUB  -1  1st time step
0.          TMOVEUB  -2  2nd time step
0.          TMOVEUB  -3  3d  time step
0.          TMOVEUB  -4  4th time step
0.          AMOVEUB  -1  1st value of amplitude
0.          AMOVEUB  -2  2nd value of amplitude
0.          AMOVEUB  -3  3d  value of amplitude
0.          AMOVEUB  -4  4th value of amplitude
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
NB=2, lev=1  ( yaw     )
Rotation
1           IMOVEAB  type of movement
3           NAXISA   =1,2,3 axis of rotation
-0.000001   TMOVEAB  -1  1st time step
10.         TMOVEAB  -2  2nd time step
0.          TMOVEAB  -3  3d  time step
0.          TMOVEAB  -4  4th time step
   5.0      AMOVEAB  -1  1st value of amplitude  (Initial MR phase + 0)
   5.0      AMOVEAB  -2  2nd value of amplitude          
0.          AMOVEAB  -3  3d  value of amplitude
0.          AMOVEAB  -4  4th value of amplitude
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
Translation
0           IMOVEUB  type of movement
3           NAXISU   =1,2,3 axis of translation
0.          TMOVEUB  -1  1st time step
0.          TMOVEUB  -2  2nd time step
0.          TMOVEUB  -3  3d  time step
0.          TMOVEUB  -4  4th time step
0.          AMOVEUB  -1  1st value of amplitude
0.          AMOVEUB  -2  2nd value of amplitude
0.          AMOVEUB  -3  3d  value of amplitude
0.          AMOVEUB  -4  4th value of amplitude
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
-----<end of movement data>----------------------------------------------------
               <blank>
Cl, Cd data / IYNVCR(.)=0 then Cl=1., Cd=0.
1           IYNVCR(1)
name.cld     FLCLCD      file name wherefrom Cl, Cd are read
               <blank>
Give the file name for the geometrical distributions
Rwing.bld 
               <blank>
Body Number   NB = 7
               <blank>
2           NLIFT   
0           IYNELSTB   
3           NBAER2ELST 
9           NNBB    
7           NCWB    
2           ISUBSCB
2
3           NLEVELSB
1           IYNTIPS 
0           IYNLES  
0           NELES   
0           IYNCONTW
3           IDIRMOB  direction for the torque calculation
               <blank>
4           LEVEL  the level of movement
               <blank>
Give  data for every level
NB=3, lev=3  (pitch )
Rotation
1           IMOVEAB  type of movement
2           NAXISA   =1,2,3 axis of rotation
-0.000001   TMOVEAB  -1  1st time step
100.        TMOVEAB  -2  2nd time step
0.          TMOVEAB  -3  3d  time step
0.          TMOVEAB  -4  4th time step!---->omega
   2.0      AMOVEAB  -1  1st value of amplitude
   2.0      AMOVEAB  -2  2nd value of amplitude
0.          AMOVEAB  -3  3d  value of amplitude
0.          AMOVEAB  -4  4th value of amplitude!---->phase
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
Translation
0           IMOVEUB  type of movement
2           NAXISU   =1,2,3 axis of translation
-0.000001   TMOVEUB  -1  1st time step
10.         TMOVEUB  -2  2nd time step
0.          TMOVEUB  -3  3d  time step
0.          TMOVEUB  -4  4th time step
   0.       AMOVEUB  -1  1st value of amplitude
   0.       AMOVEUB  -2  2nd value of amplitude
0.          AMOVEUB  -3  3d  value of amplitude
0.          AMOVEUB  -4  4th value of amplitude
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
NB=3, lev=2  (roll)
Rotation
1           IMOVEAB  type of movement
1           NAXISA   =1,2,3 axis of rotation
-0.000001   TMOVEAB  -1  1st time step
10.         TMOVEAB  -2  2nd time step
0.          TMOVEAB  -3  3d  time step
0.          TMOVEAB  -4  4th time step
  1.5       AMOVEAB  -1  1st value of amplitude
  1.5       AMOVEAB  -2  2nd value of amplitude
0.          AMOVEAB  -3  3d  value of amplitude
0.          AMOVEAB  -4  4th value of amplitude
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
Translation
0           IMOVEUB  type of movement
1           NAXISU   =1,2,3 axis of translation
0.          TMOVEUB  -1  1st time step
0.          TMOVEUB  -2  2nd time step
0.          TMOVEUB  -3  3d  time step
0.          TMOVEUB  -4  4th time step
0.          AMOVEUB  -1  1st value of amplitude
0.          AMOVEUB  -2  2nd value of amplitude
0.          AMOVEUB  -3  3d  value of amplitude
0.          AMOVEUB  -4  4th value of amplitude
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
NB=3, lev=1  ( yaw     )
Rotation
1           IMOVEAB  type of movement
3           NAXISA   =1,2,3 axis of rotation
-0.000001   TMOVEAB  -1  1st time step
10.         TMOVEAB  -2  2nd time step
0.          TMOVEAB  -3  3d  time step
0.          TMOVEAB  -4  4th time step
   5.0      AMOVEAB  -1  1st value of amplitude  (Initial MR phase + 0)
   5.0      AMOVEAB  -2  2nd value of amplitude          
0.          AMOVEAB  -3  3d  value of amplitude
0.          AMOVEAB  -4  4th value of amplitude
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
Translation
0           IMOVEUB  type of movement
3           NAXISU   =1,2,3 axis of translation
0.          TMOVEUB  -1  1st time step
0.          TMOVEUB  -2  2nd time step
0.          TMOVEUB  -3  3d  time step
0.          TMOVEUB  -4  4th time step
0.          AMOVEUB  -1  1st value of amplitude
0.          AMOVEUB  -2  2nd value of amplitude
0.          AMOVEUB  -3  3d  value of amplitude
0.          AMOVEUB  -4  4th value of amplitude
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
-----<end of movement data>----------------------------------------------------
               <blank>
Cl, Cd data / IYNVCR(.)=0 then Cl=1., Cd=0.
1           IYNVCR(1)
name.cld     FLCLCD      file name wherefrom Cl, Cd are read
               <blank>
Give the file name for the geometrical distributions
Ltail.bld 
               <blank>
Body Number   NB = 4
               <blank>
2           NLIFT   
0           IYNELSTB   
4           NBAER2ELST 
9           NNBB    
7           NCWB    
2           ISUBSCB
2
3           NLEVELSB
1           IYNTIPS 
0           IYNLES  
0           NELES   
0           IYNCONTW
3           IDIRMOB  direction for the torque calculation
               <blank>
4           LEVEL  the level of movement
               <blank>
Give  data for every level
NB=4, lev=3  (pitch )
Rotation
1           IMOVEAB  type of movement
2           NAXISA   =1,2,3 axis of rotation
-0.000001   TMOVEAB  -1  1st time step
100.        TMOVEAB  -2  2nd time step
0.          TMOVEAB  -3  3d  time step
0.          TMOVEAB  -4  4th time step!---->omega
   2.0      AMOVEAB  -1  1st value of amplitude
   2.0      AMOVEAB  -2  2nd value of amplitude
0.          AMOVEAB  -3  3d  value of amplitude
0.          AMOVEAB  -4  4th value of amplitude!---->phase
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
Translation
0           IMOVEUB  type of movement
2           NAXISU   =1,2,3 axis of translation
-0.000001   TMOVEUB  -1  1st time step
10.         TMOVEUB  -2  2nd time step
0.          TMOVEUB  -3  3d  time step
0.          TMOVEUB  -4  4th time step
   0.       AMOVEUB  -1  1st value of amplitude
   0.       AMOVEUB  -2  2nd value of amplitude
0.          AMOVEUB  -3  3d  value of amplitude
0.          AMOVEUB  -4  4th value of amplitude
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
NB=4, lev=2  (roll)
Rotation
1           IMOVEAB  type of movement
1           NAXISA   =1,2,3 axis of rotation
-0.000001   TMOVEAB  -1  1st time step
10.         TMOVEAB  -2  2nd time step
0.          TMOVEAB  -3  3d  time step
0.          TMOVEAB  -4  4th time step
  1.5       AMOVEAB  -1  1st value of amplitude
  1.5       AMOVEAB  -2  2nd value of amplitude
0.          AMOVEAB  -3  3d  value of amplitude
0.          AMOVEAB  -4  4th value of amplitude
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
Translation
0           IMOVEUB  type of movement
1           NAXISU   =1,2,3 axis of translation
0.          TMOVEUB  -1  1st time step
0.          TMOVEUB  -2  2nd time step
0.          TMOVEUB  -3  3d  time step
0.          TMOVEUB  -4  4th time step
0.          AMOVEUB  -1  1st value of amplitude
0.          AMOVEUB  -2  2nd value of amplitude
0.          AMOVEUB  -3  3d  value of amplitude
0.          AMOVEUB  -4  4th value of amplitude
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
NB=4, lev=1  ( yaw     )
Rotation
1           IMOVEAB  type of movement
3           NAXISA   =1,2,3 axis of rotation
-0.000001   TMOVEAB  -1  1st time step
10.         TMOVEAB  -2  2nd time step
0.          TMOVEAB  -3  3d  time step
0.          TMOVEAB  -4  4th time step
   5.0      AMOVEAB  -1  1st value of amplitude  (Initial MR phase + 0)
   5.0      AMOVEAB  -2  2nd value of amplitude          
0.          AMOVEAB  -3  3d  value of amplitude
0.          AMOVEAB  -4  4th value of amplitude
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
Translation
0           IMOVEUB  type of movement
3           NAXISU   =1,2,3 axis of translation
0.          TMOVEUB  -1  1st time step
0.          TMOVEUB  -2  2nd time step
0.          TMOVEUB  -3  3d  time step
0.          TMOVEUB  -4  4th time step
0.          AMOVEUB  -1  1st value of amplitude
0.          AMOVEUB  -2  2nd value of amplitude
0.          AMOVEUB  -3  3d  value of amplitude
0.          AMOVEUB  -4  4th value of amplitude
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
-----<end of movement data>----------------------------------------------------
               <blank>
Cl, Cd data / IYNVCR(.)=0 then Cl=1., Cd=0.
1           IYNVCR(1)
name.cld     FLCLCD      file name wherefrom Cl, Cd are read
               <blank>
Give the file name for the geometrical distributions
Rtail.bld 
               <blank>
Body Number   NB = 5
               <blank>
2           NLIFT   
0           IYNELSTB   
5           NBAER2ELST 
9           NNBB    
10          NCWB    
2           ISUBSCB
2
3           NLEVELSB
1           IYNTIPS 
0           IYNLES  
0           NELES   
0           IYNCONTW
3           IDIRMOB  direction for the torque calculation
               <blank>
4           LEVEL  the level of movement
               <blank>
Give  data for every level
NB=5, lev=3  (pitch )
Rotation
1           IMOVEAB  type of movement
2           NAXISA   =1,2,3 axis of rotation
-0.000001   TMOVEAB  -1  1st time step
100.        TMOVEAB  -2  2nd time step
0.          TMOVEAB  -3  3d  time step
0.          TMOVEAB  -4  4th time step!---->omega
   2.0      AMOVEAB  -1  1st value of amplitude
   2.0      AMOVEAB  -2  2nd value of amplitude
0.          AMOVEAB  -3  3d  value of amplitude
0.          AMOVEAB  -4  4th value of amplitude!---->phase
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
Translation
0           IMOVEUB  type of movement
2           NAXISU   =1,2,3 axis of translation
-0.000001   TMOVEUB  -1  1st time step
10.         TMOVEUB  -2  2nd time step
0.          TMOVEUB  -3  3d  time step
0.          TMOVEUB  -4  4th time step
   0.       AMOVEUB  -1  1st value of amplitude
   0.       AMOVEUB  -2  2nd value of amplitude
0.          AMOVEUB  -3  3d  value of amplitude
0.          AMOVEUB  -4  4th value of amplitude
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
NB=5, lev=2  (roll)
Rotation
1           IMOVEAB  type of movement
1           NAXISA   =1,2,3 axis of rotation
-0.000001   TMOVEAB  -1  1st time step
10.         TMOVEAB  -2  2nd time step
0.          TMOVEAB  -3  3d  time step
0.          TMOVEAB  -4  4th time step
  1.5       AMOVEAB  -1  1st value of amplitude
  1.5       AMOVEAB  -2  2nd value of amplitude
0.          AMOVEAB  -3  3d  value of amplitude
0.          AMOVEAB  -4  4th value of amplitude
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
Translation
0           IMOVEUB  type of movement
1           NAXISU   =1,2,3 axis of translation
0.          TMOVEUB  -1  1st time step
0.          TMOVEUB  -2  2nd time step
0.          TMOVEUB  -3  3d  time step
0.          TMOVEUB  -4  4th time step
0.          AMOVEUB  -1  1st value of amplitude
0.          AMOVEUB  -2  2nd value of amplitude
0.          AMOVEUB  -3  3d  value of amplitude
0.          AMOVEUB  -4  4th value of amplitude
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
NB=5, lev=1  ( yaw     )
Rotation
1           IMOVEAB  type of movement
3           NAXISA   =1,2,3 axis of rotation
-0.000001   TMOVEAB  -1  1st time step
10.         TMOVEAB  -2  2nd time step
0.          TMOVEAB  -3  3d  time step
0.          TMOVEAB  -4  4th time step
   5.0      AMOVEAB  -1  1st value of amplitude  (Initial MR phase + 0)
   5.0      AMOVEAB  -2  2nd value of amplitude          
0.          AMOVEAB  -3  3d  value of amplitude
0.          AMOVEAB  -4  4th value of amplitude
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
Translation
0           IMOVEUB  type of movement
3           NAXISU   =1,2,3 axis of translation
0.          TMOVEUB  -1  1st time step
0.          TMOVEUB  -2  2nd time step
0.          TMOVEUB  -3  3d  time step
0.          TMOVEUB  -4  4th time step
0.          AMOVEUB  -1  1st value of amplitude
0.          AMOVEUB  -2  2nd value of amplitude
0.          AMOVEUB  -3  3d  value of amplitude
0.          AMOVEUB  -4  4th value of amplitude
            FILTMSA  file name for TIME SERIES [IMOVEB=6]
-----<end of movement data>----------------------------------------------------
               <blank>
Cl, Cd data / IYNVCR(.)=0 then Cl=1., Cd=0.
1           IYNVCR(1)
name.cld     FLCLCD      file name wherefrom Cl, Cd are read
               <blank>
Give the file name for the geometrical distributions
rudder.bld
               <blank>



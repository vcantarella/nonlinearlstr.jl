      SUBROUTINE ELFUN ( FUVALS, XVALUE, EPVALU, NCALCF, ITYPEE, 
     *                   ISTAEV, IELVAR, INTVAR, ISTADH, ISTEPA, 
     *                   ICALCF, LTYPEE, LSTAEV, LELVAR, LNTVAR, 
     *                   LSTADH, LSTEPA, LCALCF, LFVALU, LXVALU, 
     *                   LEPVLU, IFFLAG, IFSTAT )
      INTEGER NCALCF, IFFLAG, LTYPEE, LSTAEV, LELVAR, LNTVAR
      INTEGER LSTADH, LSTEPA, LCALCF, LFVALU, LXVALU, LEPVLU
      INTEGER IFSTAT
      INTEGER ITYPEE(LTYPEE), ISTAEV(LSTAEV), IELVAR(LELVAR)
      INTEGER INTVAR(LNTVAR), ISTADH(LSTADH), ISTEPA(LSTEPA)
      INTEGER ICALCF(LCALCF)
      DOUBLE PRECISION FUVALS(LFVALU), XVALUE(LXVALU), EPVALU(LEPVLU)
C
C  Problem name : LUKSAN15  
C
C  -- produced by SIFdecode 2.4
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION X1    , X2    , X3    , X4    , P2OL  
      DOUBLE PRECISION PLI   , P     , SIGNP , PX1   , PX2   
      DOUBLE PRECISION PX3   , PX4   , PX1X1 , PX1X2 , PX1X3 
      DOUBLE PRECISION PX1X4 , PX2X2 , PX2X3 , PX2X4 , PX3X3 
      DOUBLE PRECISION PX3X4 , PX4X4 , F     , G     , H     
      LOGICAL PPOS  
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : SIGNOM    
C
       X1     = XVALUE(IELVAR(ILSTRT+     1))
       X2     = XVALUE(IELVAR(ILSTRT+     2))
       X3     = XVALUE(IELVAR(ILSTRT+     3))
       X4     = XVALUE(IELVAR(ILSTRT+     4))
       P2OL   = EPVALU(IPSTRT+     1)
       PLI    = EPVALU(IPSTRT+     2)
       P      = X1 * ( X2 ** 2 ) * ( X3 ** 3 )           
     *             * ( X4 ** 4 )                         
       PPOS   = P > 0.0D0                                
       IF (PPOS  ) SIGNP  = 1.0D0                                    
       IF (.NOT.PPOS  ) SIGNP =- 1.0D0                                  
       P      = P * SIGNP                                
       PX1    = ( X2 ** 2 ) * ( X3 ** 3 ) * ( X4 ** 4 )  
       PX2    = X1 * ( 2.0D0 * X2 ) * ( X3 ** 3 )        
     *             * ( X4 ** 4 )                         
       PX3    = X1 * ( X2 ** 2 ) * ( 3.0D0 * X3 ** 2 )   
     *             * ( X4 ** 4 )                         
       PX4    = X1 * ( X2 ** 2 ) * ( X3 ** 3 )           
     *             * ( 4.0D0 * X4 ** 3 )                 
       PX1X1  = 0.0D0                                    
       PX1X2  = ( 2.0D0 * X2 ) * ( X3 ** 3 )             
     *            * ( X4 ** 4 )                          
       PX1X3  = ( X2 ** 2 ) * ( 3.0D0 * X3 ** 2 )        
     *            * ( X4 ** 4 )                          
       PX1X4  = ( X2 ** 2 ) * ( X3 ** 3 )                
     *            * ( 4.0D0 * X4 ** 3 )                  
       PX2X2  = X1 * ( 2.0D0 ) * ( X3 ** 3 )             
     *             * ( X4 ** 4 )                         
       PX2X3  = X1 * ( 2.0D0 * X2 ) * ( 3.0D0 * X3 ** 2 )
     *             * ( X4 ** 4 )                         
       PX2X4  = X1 * ( 2.0D0 * X2 ) * ( X3 ** 3 )        
     *             * ( 4.0D0 * X4 ** 3 )                 
       PX3X3  = X1 * ( X2 ** 2 ) * ( 6.0D0 * X3 )        
     *             * ( X4 ** 4 )                         
       PX3X4  = X1 * ( X2 ** 2 ) * ( 3.0D0 * X3 ** 2 )   
     *             * ( 4.0D0 * X4 ** 3 )                 
       PX4X4  = X1 * ( X2 ** 2 ) * ( X3 ** 3 )           
     *             * ( 12.0D0 * X4 ** 2 )                
       F      = P2OL * P ** PLI                          
       G      = P2OL * PLI * P ** ( PLI - 1.0D0 )        
       H      = P2OL * PLI * ( PLI - 1.0D0 )             
     *           * P ** ( PLI - 2.0D0 )                  
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= F                                        
       ELSE
        FUVALS(IGSTRT+     1)= G * PX1                                  
        FUVALS(IGSTRT+     2)= G * PX2                                  
        FUVALS(IGSTRT+     3)= G * PX3                                  
        FUVALS(IGSTRT+     4)= G * PX4                                  
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=H * PX1 * PX1 + G * PX1X1                
         FUVALS(IHSTRT+     2)=H * PX1 * PX2 + G * PX1X2                
         FUVALS(IHSTRT+     4)=H * PX1 * PX3 + G * PX1X3                
         FUVALS(IHSTRT+     7)=H * PX1 * PX4 + G * PX1X4                
         FUVALS(IHSTRT+     3)=H * PX2 * PX2 + G * PX2X2                
         FUVALS(IHSTRT+     5)=H * PX2 * PX3 + G * PX2X3                
         FUVALS(IHSTRT+     8)=H * PX2 * PX4 + G * PX2X4                
         FUVALS(IHSTRT+     6)=H * PX3 * PX3 + G * PX3X3                
         FUVALS(IHSTRT+     9)=H * PX3 * PX4 + G * PX3X4                
         FUVALS(IHSTRT+    10)=H * PX4 * PX4 + G * PX4X4                
        END IF
       END IF
    2 CONTINUE
      RETURN
      END

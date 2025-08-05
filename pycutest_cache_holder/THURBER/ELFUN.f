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
C  Problem name : THURBER   
C
C  -- produced by SIFdecode 2.4
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION V1    , V2    , V3    , V4    , V5    
      DOUBLE PRECISION V6    , V7    , X     , E     , T     
      DOUBLE PRECISION D     , D2    , TD3   , X2    , X3    
      DOUBLE PRECISION X4    , X5    , X6    
      INTRINSIC EXP   
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : E19       
C
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       V3     = XVALUE(IELVAR(ILSTRT+     3))
       V4     = XVALUE(IELVAR(ILSTRT+     4))
       V5     = XVALUE(IELVAR(ILSTRT+     5))
       V6     = XVALUE(IELVAR(ILSTRT+     6))
       V7     = XVALUE(IELVAR(ILSTRT+     7))
       X      = EPVALU(IPSTRT+     1)
       X2     = X * X                                    
       X3     = X2 * X                                   
       X4     = X3 * X                                   
       X5     = X4 * X                                   
       X6     = X5 * X                                   
       T      = V1 + V2 * X + V3 * X2 + V4 * X3          
       D      = 1.0D0 + V5 * X + V6 * X2 + V7 * X3       
       D2     = D * D                                    
       TD3    = 0.5D0 * D2 * D                           
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= T / D                                    
       ELSE
        FUVALS(IGSTRT+     1)= 1.0D0 / D                                
        FUVALS(IGSTRT+     2)= X / D                                    
        FUVALS(IGSTRT+     3)= X2 / D                                   
        FUVALS(IGSTRT+     4)= X3 / D                                   
        FUVALS(IGSTRT+     5)= - X * T / D2                             
        FUVALS(IGSTRT+     6)= - X2 * T / D2                            
        FUVALS(IGSTRT+     7)= - X3 * T / D2                            
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+    11)=- X / D2                                 
         FUVALS(IHSTRT+    16)=- X2 / D2                                
         FUVALS(IHSTRT+    22)=- X3 / D2                                
         FUVALS(IHSTRT+    12)=- X2 / D2                                
         FUVALS(IHSTRT+    17)=- X3 / D2                                
         FUVALS(IHSTRT+    23)=- X4 / D2                                
         FUVALS(IHSTRT+    13)=- X3 / D2                                
         FUVALS(IHSTRT+    18)=- X4 / D2                                
         FUVALS(IHSTRT+    24)=- X5 / D2                                
         FUVALS(IHSTRT+    14)=- X4 / D2                                
         FUVALS(IHSTRT+    19)=- X5 / D2                                
         FUVALS(IHSTRT+    25)=- X6 / D2                                
         FUVALS(IHSTRT+    15)=X2 * T / TD3                             
         FUVALS(IHSTRT+    20)=X3 * T / TD3                             
         FUVALS(IHSTRT+    26)=X4 * T / TD3                             
         FUVALS(IHSTRT+    21)=X4 * T / TD3                             
         FUVALS(IHSTRT+    27)=X5 * T / TD3                             
         FUVALS(IHSTRT+    28)=X6 * T / TD3                             
         FUVALS(IHSTRT+     1)=0.0D+0
         FUVALS(IHSTRT+     2)=0.0D+0
         FUVALS(IHSTRT+     3)=0.0D+0
         FUVALS(IHSTRT+     4)=0.0D+0
         FUVALS(IHSTRT+     5)=0.0D+0
         FUVALS(IHSTRT+     6)=0.0D+0
         FUVALS(IHSTRT+     7)=0.0D+0
         FUVALS(IHSTRT+     8)=0.0D+0
         FUVALS(IHSTRT+     9)=0.0D+0
         FUVALS(IHSTRT+    10)=0.0D+0
        END IF
       END IF
    2 CONTINUE
      RETURN
      END

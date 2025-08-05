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
C  Problem name : MGH09     
C
C  -- produced by SIFdecode 2.4
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION V1    , V2    , V3    , V4    , X     
      DOUBLE PRECISION X2    , T     , B     , B2    , B3    
      DOUBLE PRECISION V1X   , V1X2  , V1T   , V1XT  
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : E10       
C
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       V3     = XVALUE(IELVAR(ILSTRT+     3))
       V4     = XVALUE(IELVAR(ILSTRT+     4))
       X      = EPVALU(IPSTRT+     1)
       X2     = X * X                                    
       T      = V2 * X + X2                              
       B      = V4 + V3 * X + X2                         
       B2     = B * B                                    
       B3     = B * B2                                   
       V1X    = V1 * X                                   
       V1X2   = V1 * X2                                  
       V1T    = V1 * T                                   
       V1XT   = V1X * T                                  
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= V1T / B                                  
       ELSE
        FUVALS(IGSTRT+     1)= T / B                                    
        FUVALS(IGSTRT+     2)= V1X / B                                  
        FUVALS(IGSTRT+     3)= - V1XT / B2                              
        FUVALS(IGSTRT+     4)= - V1T / B2                               
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=X / B                                    
         FUVALS(IHSTRT+     4)=- X * T / B2                             
         FUVALS(IHSTRT+     7)=- T / B2                                 
         FUVALS(IHSTRT+     5)=- V1X2 / B2                              
         FUVALS(IHSTRT+     8)=- V1X / B2                               
         FUVALS(IHSTRT+     6)=2.0 * V1X2 * T / B3                      
         FUVALS(IHSTRT+     9)=2.0 * V1XT / B3                          
         FUVALS(IHSTRT+    10)=2.0 * V1T / B3                           
         FUVALS(IHSTRT+     1)=0.0D+0
         FUVALS(IHSTRT+     3)=0.0D+0
        END IF
       END IF
    2 CONTINUE
      RETURN
      END

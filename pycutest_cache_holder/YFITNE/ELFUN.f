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
C  Problem name : YFITNE    
C
C  -- produced by SIFdecode 2.4
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION a1    , b1    , d1    , point , count 
      DOUBLE PRECISION ttan  , tsec  , tsec2 , frac  
      INTRINSIC tan   , cos   
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : tanab     
C
       a1     = XVALUE(IELVAR(ILSTRT+     1))
       b1     = XVALUE(IELVAR(ILSTRT+     2))
       d1     = XVALUE(IELVAR(ILSTRT+     3))
       point  = EPVALU(IPSTRT+     1)
       count  = EPVALU(IPSTRT+     2)
       frac   = point/count                              
       ttan   = tan(a1*(1.0-frac)+b1*frac)               
       tsec   = 1.0/cos(a1*(1.0-frac)+b1*frac)           
       tsec2  = tsec*tsec                                
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= d1*ttan                                  
       ELSE
        FUVALS(IGSTRT+     1)= d1*(1.0-frac)*tsec2                      
        FUVALS(IGSTRT+     2)= d1*frac*tsec2                            
        FUVALS(IGSTRT+     3)= ttan                                     
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=d1*((1.0-frac)**2)*tsec2*ttan            
         FUVALS(IHSTRT+     3)=d1*(frac**2)*tsec2*ttan                  
         FUVALS(IHSTRT+     2)=d1*(1.0-frac)*frac*tsec2*ttan            
         FUVALS(IHSTRT+     4)=(1.0-frac)*tsec2                         
         FUVALS(IHSTRT+     5)=frac*tsec2                               
         FUVALS(IHSTRT+     6)=0.0                                      
        END IF
       END IF
    2 CONTINUE
      RETURN
      END

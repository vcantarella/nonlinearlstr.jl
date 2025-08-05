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
C  Problem name : ROSZMAN1  
C
C  -- produced by SIFdecode 2.4
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION V1    , V2    , X     , PI    , PIR   
      DOUBLE PRECISION PIR2  , V12   , V13   , V2MX  , V2MX2 
      DOUBLE PRECISION V2MX3 , R     
      INTRINSIC ATAN  
      IFSTAT = 0
      PI     = 4.0 * ATAN( 1.0 )                        
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : E7        
C
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       X      = EPVALU(IPSTRT+     1)
       V12    = V1 * V1                                  
       V13    = V1 * V12                                 
       V2MX   = V2 - X                                   
       V2MX2  = V2MX * V2MX                              
       V2MX3  = V2MX * V2MX2                             
       R      = V12 / V2MX2 + 1.0                        
       PIR    = PI * R                                   
       PIR2   = PIR * R                                  
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= - ATAN( V1 / V2MX ) / PI                 
       ELSE
        FUVALS(IGSTRT+     1)= - 1.0 / ( PIR * V2MX )                   
        FUVALS(IGSTRT+     2)= V1 / ( PIR * V2MX2 )                     
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=2.0 * V1 / ( PIR2 * V2MX3 )              
         FUVALS(IHSTRT+     2)=1.0 / ( PIR * V2MX2 )                    
     *                         - 2.0 * V12 / ( PIR2 * V2MX ** 4 )       
         FUVALS(IHSTRT+     3)=2.0 * V13 / ( PIR2 * V2MX ** 5 )         
     *                         - 2.0 * V1 / ( PIR * V2MX3 )             
        END IF
       END IF
    2 CONTINUE
      RETURN
      END

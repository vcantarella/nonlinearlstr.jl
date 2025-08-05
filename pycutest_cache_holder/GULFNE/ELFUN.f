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
C  Problem name : GULFNE    
C
C  -- produced by SIFdecode 2.4
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION V1    , V2    , V3    , T     , V1SQ  
      DOUBLE PRECISION YMV2  , YMV2SQ, LNYMV2, A     , ALN   
      DOUBLE PRECISION AM1   , EXPMA , AEXPMA
      INTRINSIC LOG   , EXP   , ABS   
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : GLF       
C
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       V3     = XVALUE(IELVAR(ILSTRT+     3))
       T      = EPVALU(IPSTRT+     1)
       V1SQ   = V1 * V1                                  
       YMV2   = 25.0 + ( -50.0 * LOG( T ) )**(2.0/3.0)   
     *          - V2                                     
       YMV2SQ = YMV2 * YMV2                              
       LNYMV2 = LOG( ABS( YMV2 ) )                       
       A      = ABS( YMV2 )**V3 / V1                     
       AM1    = A - 1.0                                  
       ALN    = A * LNYMV2                               
       EXPMA  = EXP( - A )                               
       AEXPMA = A * EXPMA                                
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= EXPMA                                    
       ELSE
        FUVALS(IGSTRT+     1)= AEXPMA / V1                              
        FUVALS(IGSTRT+     2)= V3 * AEXPMA / YMV2                       
        FUVALS(IGSTRT+     3)= - AEXPMA * LNYMV2                        
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=( A - 2.0 ) * AEXPMA / V1SQ              
         FUVALS(IHSTRT+     2)=V3 * AM1 * AEXPMA / ( V1 * YMV2 )        
         FUVALS(IHSTRT+     4)=- ALN * AEXPMA /   V1                    
         FUVALS(IHSTRT+     3)=V3 * AEXPMA * (1.0 + V3 * AM1 ) / YMV2SQ 
         FUVALS(IHSTRT+     5)=AEXPMA * ( 1.0 + V3 * ALN )  /   YMV2    
         FUVALS(IHSTRT+     6)=ALN * LNYMV2 * EXPMA * AM1               
        END IF
       END IF
    2 CONTINUE
      RETURN
      END

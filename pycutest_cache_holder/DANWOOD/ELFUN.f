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
C  Problem name : DANWOOD   
C
C  -- produced by SIFdecode 2.4
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION V1    , V2    , X     , V1X   , V2X   
      DOUBLE PRECISION LOGV1X, V1XV2 , V1XV21
      INTRINSIC LOG   
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : E1        
C
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       X      = EPVALU(IPSTRT+     1)
       V1X    = V1 * X                                   
       V2X    = V2 * X                                   
       LOGV1X = LOG( V1X )                               
       V1XV2  = V1X ** V2                                
       V1XV21 = V1X ** ( V2 - 1.0 )                      
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= V1XV2                                    
       ELSE
        FUVALS(IGSTRT+     1)= V2X * V1XV21                             
        FUVALS(IGSTRT+     2)= LOGV1X * V1XV2                           
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=V2 * ( V2 - 1.0 ) * X ** 2               
     *                          * V1X ** ( V2 - 2.0 )                   
         FUVALS(IHSTRT+     2)=X * V1XV21 + V2X * LOGV1X * V1XV21       
         FUVALS(IHSTRT+     3)=V1XV2 * LOGV1X ** 2                      
        END IF
       END IF
    2 CONTINUE
      RETURN
      END
